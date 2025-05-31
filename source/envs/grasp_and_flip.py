import os
import numpy as np
from dataclasses import dataclass

from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import spawn_usd, UsdFileCfg
from isaaclab.utils.math import quat_mul
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg

from source.utils.reward import sparse_grasp_flip_reward

@dataclass
class GraspAndFlipEnvCfg:
    table_height: float = 0.295  # Match your table USD height
    flip_axis: list[float] = [0.0, 0.0, 1.0]  # Z-axis flip
    flip_angle: float = np.pi  # 180 degree flip
    pos_tol: float = 0.02  # 2cm tolerance
    ori_tol: float = 0.1   # ~6 degree tolerance
    grasp_angle: float = 1.0  # finger close target (radians)

class GraspAndFlipEnv(DirectRLEnv):
    def __init__(self, env_cfg, task_cfg: GraspAndFlipEnvCfg):
        self.task_cfg = task_cfg
        super().__init__(cfg=env_cfg)
        
    def _setup_scene(self):
        """Setup the scene with table, cube, and hand."""
        cwd = os.path.dirname(__file__) + "/../../assets"
        
        # 1) Table at hand height (0.295m from your USD)
        table_usd = os.path.join(cwd, "table_at_hand_height.usd")
        spawn_usd(
            cfg=UsdFileCfg(
                usd_path=table_usd,
                prim_path="/World/Table"
            )
        )
        
        # 2) Cube on table - positioned for optimal hand reach
        cube_usd = os.path.join(cwd, "cube.usd")
        cube_height = 0.025  # Half of cube size (5cm cube)
        spawn_usd(
            cfg=UsdFileCfg(
                usd_path=cube_usd,
                prim_path="/World/Cube",
                translation=[0.0, -0.15, self.task_cfg.table_height + cube_height]  # Optimal reach distance
            )
        )
        
        # 3) Shadow Hand positioned for optimal reach
        # Shadow Hand dimensions: ~20cm length, needs to be positioned appropriately
        hand_cfg = ArticulationCfg(
            spawn=UsdFileCfg(
                usd_path=f"{os.getenv('ISAAC_NUCLEUS_DIR')}/Props/ShadowHand/shadow_hand_instanceable.usd",
                prim_path="/World/Hand",
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0.0, 0.0, self.task_cfg.table_height + 0.20],  # 20cm above table
                rot=[0.7071, 0.0, 0.0, 0.7071],  # Rotated to face down toward cube
            ),
        )
        self.hand = Articulation(cfg=hand_cfg)
        
        # 4) Frame transformer for poses
        frame_cfg = FrameTransformerCfg(
            prim_path="/World/Cube",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/Cube",
                    name="cube_frame",
                ),
            ],
        )
        self.cube_frame = FrameTransformer(cfg=frame_cfg)

    def _get_observations(self) -> dict:
        """Get current observations."""
        # Hand joint positions
        joint_pos = self.hand.data.joint_pos[0]  # First environment
        
        # Cube pose
        cube_pos = self.cube_frame.data.target_pos_w[0, 0]  # First env, first frame
        cube_rot = self.cube_frame.data.target_quat_w[0, 0]  # First env, first frame
        
        return {
            "obs": np.concatenate([joint_pos, cube_pos, cube_rot]), 
            "achieved_goal": {"position": cube_pos, "orientation": cube_rot},
            "desired_goal": self.current_goal 
        }

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        obs_dict = self._get_observations()
        return sparse_grasp_flip_reward(
            obs_dict["achieved_goal"],
            obs_dict["desired_goal"],
            self.task_cfg.pos_tol,
            self.task_cfg.ori_tol
        )

    def _reset_idx(self, env_ids):
        """Reset specific environments."""
        super()._reset_idx(env_ids)
        
        # Reset hand to initial pose
        self.hand.set_joint_position_target(
            torch.zeros_like(self.hand.data.joint_pos), 
            env_ids=env_ids
        )
        
        # Close hand to grasp cube
        self._close_hand(env_ids)
        
        # Compute flip goal from current cube orientation
        cube_pos = self.cube_frame.data.target_pos_w[env_ids, 0]
        cube_rot = self.cube_frame.data.target_quat_w[env_ids, 0]
        
        # Calculate desired orientation after flip
        axis = torch.tensor(self.task_cfg.flip_axis, device=self.device)
        angle = self.task_cfg.flip_angle
        
        # Create flip quaternion
        flip_quat = torch.zeros(4, device=self.device)
        flip_quat[3] = np.cos(angle/2)  # w
        flip_quat[:3] = axis * np.sin(angle/2)  # xyz
        
        # Apply flip to current orientation
        desired_ori = quat_mul(flip_quat.unsqueeze(0), cube_rot)
        
        self.current_goal = {
            "position": cube_pos,
            "orientation": desired_ori
        }

    def _close_hand(self, env_ids):
        """Close all hand joints to grasp."""
        # Shadow Hand has 24 joints (20 finger joints + 4 wrist joints)
        targets = torch.ones(24, device=self.device) * self.task_cfg.grasp_angle
        self.hand.set_joint_position_target(targets.unsqueeze(0), env_ids=env_ids)

    @property 
    def num_hand_joints(self):
        """Number of hand joints."""
        return self.hand.num_joints

