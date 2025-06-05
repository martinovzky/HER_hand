# her_hand/source/envs/grasp_and_flip.py

import os
import numpy as np
import torch
from dataclasses import dataclass, field

from isaaclab.envs import DirectRLEnv
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG

# Import spawn_from_usd() and its config class exactly as documented
from isaaclab.sim.spawners.from_files import spawn_from_usd, UsdFileCfg
from isaaclab.sim.spawners.shapes import spawn_cuboid, CuboidCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg, MassPropertiesCfg

from isaaclab.assets import RigidObject, RigidObjectCfg, Articulation, ArticulationCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.utils.math import quat_mul

from source.utils.reward import sparse_grasp_flip_reward

@dataclass
class GraspAndFlipEnvCfg:
    table_height: float       = 0.295            # Top of table in meters
    flip_axis:    list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0]) # Flip around Z‐axis
    flip_angle:   float       = float(np.pi)    # 180° flip
    pos_tol:      float       = 0.02             # 2 cm tolerance
    ori_tol:      float       = 0.1              # ~6° tolerance
    grasp_angle:  float       = 1.0              # Radians to fully close fingers

class GraspAndFlipEnv(DirectRLEnv):
    def __init__(self, env_cfg, task_cfg: GraspAndFlipEnvCfg):
        self.task_cfg = task_cfg
        super().__init__(cfg=env_cfg)

    def _setup_scene(self):
        """Setup scene with proper collision and physics interactions."""
        
        # === 1) Create table using built-in cuboid spawner ===
        table_cfg = RigidObjectCfg(
            prim_path=f"{self.scene.env_ns}/Table",
            spawn=CuboidCfg(
                size=(1.0, 1.0, 0.1),  # 1m x 1m x 10cm table
                rigid_props=RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.01,
                    rest_offset=0.0,
                ),
                mass_props=MassPropertiesCfg(mass=10.0),
                visual_material_path="/World/Looks/TableMaterial",
            ),
            collision_group=0,
            debug_vis=False,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.task_cfg.table_height - 0.05),  # Center table at correct height
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.table = RigidObject(cfg=table_cfg)

        # === 2) Create cube using built-in cuboid spawner ===
        cube_cfg = RigidObjectCfg(
            prim_path=f"{self.scene.env_ns}/Cube",
            spawn=CuboidCfg(
                size=(0.05, 0.05, 0.05),  # 5cm cube
                rigid_props=RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=False,
                ),
                collision_props=CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.01,
                    rest_offset=0.0,
                ),
                mass_props=MassPropertiesCfg(mass=0.1),
                visual_material_path="/World/Looks/CubeMaterial",
            ),
            collision_group=-1,
            debug_vis=False,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.task_cfg.table_height + 0.025),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.cube = RigidObject(cfg=cube_cfg)

        # === 3) Create Shadow Hand ===
        hand_cfg = ArticulationCfg(
            prim_path=f"{self.scene.env_ns}/Hand",
            spawn=SHADOW_HAND_CFG.spawn,
            collision_group=0,
            debug_vis=False,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.task_cfg.table_height + 0.20),
                rot=(0.0, 0.0, -0.7071, 0.7071),
                joint_pos={".*": 0.0},
            ),
            actuators=SHADOW_HAND_CFG.actuators,
            soft_joint_pos_limit_factor=SHADOW_HAND_CFG.soft_joint_pos_limit_factor,
        )
        self.hand = Articulation(cfg=hand_cfg)

        # === 4) FrameTransformer for cube pose ===
        frame_cfg = FrameTransformerCfg(
            prim_path=f"{self.scene.env_ns}/Cube",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{self.scene.env_ns}/Cube",
                    name="cube_frame"
                )
            ]
        )
        self.cube_frame = FrameTransformer(cfg=frame_cfg)

        # === 5) Clone environments ===
        self.scene.clone_environments(copy_from_source=False)
        
        # === 6) Filter collisions ===
        self.scene.filter_collisions(global_prim_paths=[])

        # === 7) Register all objects with scene ===
        self.scene.sensors["cube_frame"] = self.cube_frame
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["cube"] = self.cube
        self.scene.articulations["hand"] = self.hand

    def _get_observations(self) -> dict:
        """
        Called each step to collect observations:
          - 'obs': [hand_joint_positions, cube_x,y,z, cube_qx,qy,qz,qw]
          - 'achieved_goal': {'position': [x,y,z], 'orientation': [qx,qy,qz,qw]}
          - 'desired_goal':  self.current_goal (set in _reset_idx)
        """
        # Hand joint positions (first env index) - use data attribute
        joint_pos = self.hand.data.joint_pos[0]  # shape = (num_joints,)

        # Cube pose from FrameTransformer (env_id=0, frame index=0)
        cube_pos = self.cube_frame.data.target_pos_w[0, 0]    # shape = (3,)
        cube_rot = self.cube_frame.data.target_quat_w[0, 0]   # shape = (4,)

        # Flatten achieved_goal and desired_goal
        achieved_goal = np.concatenate([
            cube_pos.cpu().numpy(),
            cube_rot.cpu().numpy()
        ])  # shape = (7,)
        
        desired_goal = np.concatenate([
            self.current_goal["position"],
            self.current_goal["orientation"]
        ])  # shape = (7,)

        return {
            "observation": np.concatenate([
                joint_pos.cpu().numpy(),
                cube_pos.cpu().numpy(),
                cube_rot.cpu().numpy()
            ]),
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }

    def _get_rewards(self) -> torch.Tensor:
        """
        Called each simulation step to compute a batch of rewards (size = num_envs).
        Uses sparse binary reward: 1.0 if within both position and orientation tolerances.
        """
        od = self._get_observations()
        
        # Convert flattened goals back to dict format for reward function
        achieved = {
            "position": od["achieved_goal"][:3],
            "orientation": od["achieved_goal"][3:7]
        }
        desired = {
            "position": od["desired_goal"][:3], 
            "orientation": od["desired_goal"][3:7]
        }
        
        r = sparse_grasp_flip_reward(
            achieved,
            desired,
            self.task_cfg.pos_tol,
            self.task_cfg.ori_tol
        )
        return torch.tensor([r], device=self.device)


    # Stable-Baselines3 HER requires `compute_reward` to be implemented
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the sparse grasp-and-flip reward for HER."""
        # Convert flattened goals back to dict format
        achieved = {
            "position": achieved_goal[:3],
            "orientation": achieved_goal[3:7]
        }
        desired = {
            "position": desired_goal[:3], 
            "orientation": desired_goal[3:7]
        }
        
        return float(
            sparse_grasp_flip_reward(
                achieved,
                desired,
                self.task_cfg.pos_tol,
                self.task_cfg.ori_tol,
            )
        )
    
    def _reset_idx(self, env_ids):
        """
        Called when specific environments need resetting (new episode).
        We:
          1) super()._reset_idx resets physics state and counters
          2) reset all hand joints to zero
          3) close hand by setting all hand joints = grasp_angle
          4) read cube pose, compute flipped goal quaternion, store in current_goal
        """
        super()._reset_idx(env_ids)

        # --- 2) Reset hand joint positions to zero ---
        zero_joints = torch.zeros_like(self.hand.data.joint_pos[env_ids])
        self.hand.set_joint_position_target(zero_joints, env_ids=env_ids)

        # --- 3) Close hand (grasp) by setting each joint to grasp_angle ---
        gval   = self.task_cfg.grasp_angle
        gvec   = torch.ones(self.hand.num_joints, device=self.device) * gval
        gbatch = gvec.unsqueeze(0).repeat(len(env_ids), 1)  # shape = (num_envs, num_joints)
        self.hand.set_joint_position_target(gbatch, env_ids=env_ids)

        # --- 4) Compute flipped goal orientation based on cube's current pose ---
        cube_pos = self.cube_frame.data.target_pos_w[env_ids, 0]   # shape = (num_envs, 3)
        cube_rot = self.cube_frame.data.target_quat_w[env_ids, 0]  # shape = (num_envs, 4)

        # Convert flip_axis to float tensor with correct device
        axis = torch.tensor(self.task_cfg.flip_axis, dtype=torch.float32, device=self.device)  # shape = (3,)
        angle = self.task_cfg.flip_angle
        qw = float(np.cos(angle / 2.0))
        qxyz = (axis / torch.norm(axis)) * float(np.sin(angle / 2.0))
        flip_quat_base = torch.cat([qxyz, torch.tensor([qw], dtype=torch.float32, device=self.device)])  # shape = (4,)

        flip_quat = flip_quat_base.unsqueeze(0).repeat(len(env_ids), 1)  # shape = (num_envs, 4)
        desired_ori = quat_mul(flip_quat, cube_rot)                     # shape = (num_envs, 4)

        self.current_goal = {
            "position": cube_pos[0].cpu().numpy(),       # shape = (3,) - remove batch dimension
            "orientation": desired_ori[0].cpu().numpy()  # shape = (4,) - remove batch dimension
        }

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Called before each physics step to process actions.
        Convert actions to joint position targets for the Shadow Hand.
        """
        # Apply actions as joint position targets to the hand
        # actions should have shape (num_envs, num_joints)
        self.hand.set_joint_position_target(actions)

    def _apply_action(self) -> None:
        """
        Called after _pre_physics_step to write data to simulation.
        The joint targets are already set in _pre_physics_step, so nothing to do here.
        """
        pass

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Called each step to determine if episodes are done.
        Returns:
            terminated: Episodes that reached terminal condition
            truncated: Episodes that reached time limit
        """
        # For now, never terminate episodes early (let time limit handle it)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, truncated

    @property  
    def num_hand_joints(self) -> int:
        """Return the total number of joints in the Shadow Hand articulation."""
        return self.hand.num_joints

    def step(self, action):
        """Preprocess action and postprocess return values for SB3 compatibility."""
        # Convert numpy array to tensor if needed
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device)
        
        # Call the parent step method
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Convert CUDA tensors to CPU/NumPy for SB3 compatibility
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()
        
        # Convert any tensors in info dict to numpy
        if isinstance(info, dict):
            for key, value in info.items():
                if isinstance(value, torch.Tensor):
                    info[key] = value.cpu().numpy()
        
        return obs, reward, terminated, truncated, info
        

