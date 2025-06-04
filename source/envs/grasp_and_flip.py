# her_hand/source/envs/grasp_and_flip.py

import os
import numpy as np
import torch
from dataclasses import dataclass

from isaaclab.envs import DirectRLEnv

# Import spawn_from_usd() and its config class exactly as documented
from isaaclab.sim.spawners.from_files import spawn_from_usd, UsdFileCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg

from isaaclab.assets import RigidObject, RigidObjectCfg, Articulation, ArticulationCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.utils.math import quat_mul

from source.utils.reward import sparse_grasp_flip_reward

@dataclass
class GraspAndFlipEnvCfg:
    table_height: float       = 0.295            # Top of table in meters
    flip_axis:    list[float] = [0.0, 0.0, 1.0]  # Flip around Z‐axis
    flip_angle:   float       = float(np.pi)    # 180° flip
    pos_tol:      float       = 0.02             # 2 cm tolerance
    ori_tol:      float       = 0.1              # ~6° tolerance
    grasp_angle:  float       = 1.0              # Radians to fully close fingers

class GraspAndFlipEnv(DirectRLEnv):
    def __init__(self, env_cfg, task_cfg: GraspAndFlipEnvCfg):
        self.task_cfg = task_cfg
        super().__init__(cfg=env_cfg)

    def _setup_scene(self):
        """
        Called once when the environment initializes. We:
          1) spawn the table USD under /World/Table via spawn_from_usd
          2) wrap /World/Table in a static RigidObject
          3) spawn the cube USD under /World/Cube (positioned on table)
          4) wrap /World/Cube in a dynamic RigidObject
          5) spawn the Shadow Hand USD under /World/Hand (rotated to face downward)
          6) wrap /World/Hand in an Articulation
          7) create a FrameTransformer to read /World/Cube pose each step
        """
        # Resolve absolute path to assets directory
        here   = os.path.dirname(__file__)
        assets = os.path.abspath(os.path.join(here, "../../assets"))

        # === 1) Spawn table USD under /World/Table ===
        table_usd = os.path.join(assets, "table_at_hand_height.usd")
        spawn_from_usd(
            "/World/Table",
            UsdFileCfg(
                usd_path=table_usd,
                # Make table static by enabling kinematic (rigid_props)
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
                # Ensure collision meshes are created
                collision_props=CollisionPropertiesCfg(),
                # Keep it visible by default, no extra variants/materials
            )
        )
        # === 2) Wrap /World/Table in RigidObject ===
        table_cfg = RigidObjectCfg(
            prim_path="/World/Table",
            collision_group=0,
            debug_vis=False
        )
        self.table = RigidObject(cfg=table_cfg)

        # === 3) Spawn cube USD under /World/Cube, half‐height above table ===
        cube_usd  = os.path.join(assets, "cube.usd")
        half_edge = 0.025  # 5cm cube → half size = 0.025m
        spawn_from_usd(
            "/World/Cube",
            UsdFileCfg(
                usd_path=cube_usd,
                # Let cube be dynamic
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False),
                collision_props=CollisionPropertiesCfg(),
            )
        )
        # === 4) Wrap /World/Cube in RigidObject ===
        cube_cfg = RigidObjectCfg(
            prim_path="/World/Cube",
            collision_group=0,
            debug_vis=False,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.task_cfg.table_height + half_edge),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.cube = RigidObject(cfg=cube_cfg)

        # === 5) Spawn Shadow Hand USD under /World/Hand, rotated to face downward ===
        hand_usd = os.path.join(
            os.getenv("ISAAC_NUCLEUS_DIR", ""),
            "Props/ShadowHand/shadow_hand_instanceable.usd"
        )
        spawn_from_usd(
            "/World/Hand",
            UsdFileCfg(
                usd_path=hand_usd,
                # Enable collision on hand
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False),
                collision_props=CollisionPropertiesCfg(),
                # No variants, no material overrides
            )
        )
        # === 6) Wrap /World/Hand in Articulation ===
        hand_cfg = ArticulationCfg(
            prim_path="/World/Hand",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.task_cfg.table_height + 0.20),
                rot=(0.7071, 0.0, 0.0, 0.7071),
            ),
            collision_group=0,
            debug_vis=False,
            # default_joint_positions left as None (use USD default)
        )
        self.hand = Articulation(cfg=hand_cfg)

        # === 7) FrameTransformer to read cube pose every step ===
        frame_cfg = FrameTransformerCfg(
            prim_path="/World/Cube",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/Cube",
                    name="cube_frame"
                )
            ]
        )
        self.cube_frame = FrameTransformer(cfg=frame_cfg)

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

        return {
            "obs": np.concatenate([
                joint_pos.cpu().numpy(),
                cube_pos.cpu().numpy(),
                cube_rot.cpu().numpy()
            ]),
            "achieved_goal": {
                "position": cube_pos.cpu().numpy(),
                "orientation": cube_rot.cpu().numpy()
            },
            "desired_goal": self.current_goal
        }

    def _get_rewards(self) -> torch.Tensor:
        """
        Called each simulation step to compute a batch of rewards (size = num_envs).
        Uses sparse binary reward: 1.0 if within both position and orientation tolerances.
        """
        od = self._get_observations()
        r  = sparse_grasp_flip_reward(
            od["achieved_goal"],
            od["desired_goal"],
            self.task_cfg.pos_tol,
            self.task_cfg.ori_tol
        )
        return torch.tensor([r], device=self.device)


    # Stable-Baselines3 HER requires `compute_reward` to be implemented
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the sparse grasp-and-flip reward for HER."""
        return float(
            sparse_grasp_flip_reward(
                achieved_goal,
                desired_goal,
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

        axis  = torch.tensor(self.task_cfg.flip_axis, device=self.device)  # shape = (3,)
        angle = self.task_cfg.flip_angle
        qw    = float(np.cos(angle / 2.0))
        qxyz  = (axis / torch.norm(axis)) * float(np.sin(angle / 2.0))
        flip_quat_base = torch.cat([qxyz, torch.tensor([qw], device=self.device)])  # shape = (4,)

        flip_quat = flip_quat_base.unsqueeze(0).repeat(len(env_ids), 1)  # shape = (num_envs, 4)
        desired_ori = quat_mul(flip_quat, cube_rot)                     # shape = (num_envs, 4)

        self.current_goal = {
            "position": cube_pos.cpu().numpy(),       # shape = (num_envs, 3)
            "orientation": desired_ori.cpu().numpy()   # shape = (num_envs, 4)
        }

    @property
    def num_hand_joints(self) -> int:
        """Return the total number of joints in the Shadow Hand articulation."""
        return self.hand.num_joints

