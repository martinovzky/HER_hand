import os
import numpy as np
from dataclasses import dataclass

from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import spawn_usd, UsdFileCfg

from source.utils.reward import sparse_grasp_flip_reward

@dataclass
class GraspAndFlipEnvCfg:
    table_height: float
    flip_axis: list[float]
    flip_angle: float
    pos_tol: float
    ori_tol: float
    grasp_angle: float  # finger-close target

class GraspAndFlipEnv(DirectRLEnv):
    def __init__(self, env_cfg, task_cfg: GraspAndFlipEnvCfg):
        super().__init__(cfg=env_cfg)
        self.task_cfg = task_cfg

    def build_scene(self):
        cwd = os.path.dirname(__file__) + "/../../assets"
        # 1) Table at hand height
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
                translation=[0.0, -0.1, self.task_cfg.table_height + cube_height]  # Slightly in front of hand
            )
        )
        # 3) Shadow Hand positioned for optimal reach
        hand_usd = os.path.join(
            os.getenv("EXPASSETS_PATH", ""),
            "isaaclab_assets/robots/shadow_hand.usd"
        )
        hand_height = 0.15  # Height above table for proper grasp approach
        spawn_usd(
            cfg=UsdFileCfg(
                usd_path=hand_usd,
                prim_path="/World/Hand",
                translation=[0.0, 0.0, self.task_cfg.table_height + hand_height]
            )
        )

    def reset(self):
        obs = super().reset()
        # 1) close hand to grasp cube
        self.close_hand()
        # 2) compute flip goal from current cube orientation
        pose = self.get_world_pose("/World/Cube")
        axis  = np.array(self.task_cfg.flip_axis)
        angle = self.task_cfg.flip_angle
        qw = np.cos(angle/2)
        q_xyz = axis/np.linalg.norm(axis) * np.sin(angle/2)
        flip_q = np.concatenate([q_xyz, [qw]])
        desired_ori = quat_mult(flip_q, pose.orientation)
        self.current_goal = {
            "position": pose.position,
            "orientation": desired_ori
        }
        return obs

    def get_observations(self):
        jp = self.get_articulation_joint_positions("/World/Hand") #joint positions
        cp = self.get_world_pose("/World/Cube") # cube pose
        return {
            "obs": np.concatenate([jp, cp.position, cp.orientation]), 
            "achieved_goal": {"position": cp.position, "orientation": cp.orientation},
            "desired_goal": self.current_goal 
        }

    def compute_reward(self, obs_dict):
        return sparse_grasp_flip_reward(
            obs_dict["achieved_goal"],
            obs_dict["desired_goal"],
            self.task_cfg.pos_tol,
            self.task_cfg.ori_tol
        )

    def close_hand(self):
        """Close all hand joints to grasp."""
        num = self.num_hand_joints
        targets = np.ones(num) * self.task_cfg.grasp_angle 
        self.set_articulation_joint_targets("/World/Hand", targets)

# helper for quaternion multiplication
def quat_mult(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    return np.array([x, y, z, w])

