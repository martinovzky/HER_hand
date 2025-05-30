# her_hand/source/extension.py

import os
import yaml

# 1) Import only AppLauncher at module level; other IsaacLab imports happen inside launch()
from isaaclab.app import AppLauncher

def launch(headless: bool, record_video: bool, return_env: bool = False):
    # ------------------------------------------------------------------------------
    # 1) Start Isaac Lab via AppLauncher
    # ------------------------------------------------------------------------------
    launcher = AppLauncher(
        headless=headless,
        enable_cameras=record_video
    )
    app = launcher.app

    # ------------------------------------------------------------------------------
    # 2) Now that App is running, import the rest of the IsaacLab modules
    # ------------------------------------------------------------------------------
    from isaaclab.sim import SimulationContext
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
    from source.envs.grasp_and_flip import GraspAndFlipEnv, GraspAndFlipEnvCfg

    # ------------------------------------------------------------------------------
    # 3) Create a SimulationContext (physics + renderer). DirectRLEnv will pick it up.
    # ------------------------------------------------------------------------------
    sim = SimulationContext(
        physics_dt=1/60.0,
        rendering_dt=1/30.0
    )

    # ------------------------------------------------------------------------------
    # 4) Load your YAML configuration (source/config.yaml)
    # ------------------------------------------------------------------------------
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r') as f:
        cfg_data = yaml.safe_load(f)
    if not isinstance(cfg_data, dict):
        raise ValueError(f"Expected dict from {config_path}, got {type(cfg_data)}")

    # ------------------------------------------------------------------------------
    # 5) Build DirectRLEnvCfg with a ViewerCfg (never None)
    # ------------------------------------------------------------------------------
    viewer_cfg = ViewerCfg()  # Always supply a ViewerCfg instance
    env_cfg    = DirectRLEnvCfg(
        viewer=viewer_cfg,
        episode_length_s=10.0,
        decimation=2
    )

    # ------------------------------------------------------------------------------
    # 6) Create the task configuration and RL environment
    # ------------------------------------------------------------------------------
    env_section = cfg_data.get("env")
    if not isinstance(env_section, dict):
        raise ValueError("'env' section missing or not a dict in config.yaml")
    task_cfg = GraspAndFlipEnvCfg(**env_section)

    # Construct the environment; note that GraspAndFlipEnv signature is (env_cfg, task_cfg)
    env = GraspAndFlipEnv(env_cfg, task_cfg)

    # ------------------------------------------------------------------------------
    # 7) Reset to build the scene and get the first observation
    # ------------------------------------------------------------------------------
    obs = env.reset()

    # ------------------------------------------------------------------------------
    # 8) Simple rollout/training loop
    #    DirectRLEnv.step() returns: (obs, reward, terminated, truncated, info)
    # ------------------------------------------------------------------------------
    train_section = cfg_data.get("train", {})
    total_steps   = int(train_section.get("total_steps", 0)) if isinstance(train_section, dict) else 0

    for _ in range(total_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # If any environment is done or truncated, reset
        if terminated.any() or truncated.any():
            obs = env.reset()

    # ------------------------------------------------------------------------------
    # 9) Return or close
    # ------------------------------------------------------------------------------
    if return_env:
        return env
    else:
        env.close()
        app.close()
        return None




