# her_hand/source/extension.py

import os
import yaml

from isaaclab.app import AppLauncher               # launches Isaac Lab
from isaaclab.sim import SimulationContext         # physics + renderer controller
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg

from source.envs.grasp_and_flip import GraspAndFlipEnv, GraspAndFlipEnvCfg

def launch(headless: bool, record_video: bool, return_env: bool = False):
    # ------------------------------------------------------------------------------
    # 1) Start Isaac Lab via AppLauncher
    #    - headless: whether to disable the GUI entirely
    #    - enable_cameras: whether to initialize offscreen cameras for video recording
    # ------------------------------------------------------------------------------
    launcher = AppLauncher(
        headless=headless,
        enable_cameras=record_video
    )
    app = launcher.app

    # ------------------------------------------------------------------------------
    # 2) Create a SimulationContext (physics + renderer). We DO NOT pass this
    #    into DirectRLEnvCfg; DirectRLEnv will implicitly use the active SimulationContext.
    # ------------------------------------------------------------------------------
    sim = SimulationContext(
        physics_dt=1/60.0,
        rendering_dt=1/30.0
    )

    # ------------------------------------------------------------------------------
    # 3) Load YAML configuration from "source/config.yaml"
    #    Ensure that safe_load returned a dict (not None).
    # ------------------------------------------------------------------------------
    cfg_path = os.path.join("source", "config.yaml")
    cfg_data = yaml.safe_load(open(cfg_path, "r"))
    if not isinstance(cfg_data, dict):
        raise RuntimeError(f"Expected a dict from {cfg_path} but got {type(cfg_data)}")

    # ------------------------------------------------------------------------------
    # 4) Build DirectRLEnvCfg with a ViewerCfg
    #    - In v2.x, DirectRLEnvCfg only takes its own fields, and we must supply viewer=ViewerCfg()
    # ------------------------------------------------------------------------------
    viewer_cfg = ViewerCfg()         # Always pass a ViewerCfg instance (cannot pass None)
    env_cfg    = DirectRLEnvCfg(viewer=viewer_cfg)

    # ------------------------------------------------------------------------------
    # 5) Build the task config from cfg_data["env"] and create the GraspAndFlipEnv
    #    - Check that "env" key exists and is a dict
    # ------------------------------------------------------------------------------
    if "env" not in cfg_data or not isinstance(cfg_data["env"], dict):
        raise RuntimeError("'env' section missing or not a dict in config.yaml")
    task_cfg = GraspAndFlipEnvCfg(**cfg_data["env"])
    env      = GraspAndFlipEnv(env_cfg, task_cfg)

    # ------------------------------------------------------------------------------
    # 6) Reset the environment to build the scene and retrieve the first observation
    #    (DirectRLEnv in v2.x does not have `initialize()`, use `reset()` instead)
    # ------------------------------------------------------------------------------
    obs = env.reset()

    # ------------------------------------------------------------------------------
    # 7) Run a simple loop for cfg_data["train"]["total_steps"] steps
    #    - In v2.x, env.step(...) returns exactly five values: (obs, reward, done, truncated, info)
    # ------------------------------------------------------------------------------
    total_steps = 0
    if isinstance(cfg_data.get("train"), dict):
        total_steps = int(cfg_data["train"].get("total_steps", 0))

    for _ in range(total_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs = env.reset()

    # ------------------------------------------------------------------------------
    # 8) Either return the env, or shut down the application
    # ------------------------------------------------------------------------------
    if return_env:
        return env
    else:
        app.close()



