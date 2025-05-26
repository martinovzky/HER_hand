# source/extension.py

import yaml
from isaaclab.app import AppLauncher             # launches Isaac Lab :contentReference[oaicite:0]{index=0}
from isaaclab.sim import SimulationContext       # physics+renderer controller :contentReference[oaicite:1]{index=1}
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # RL base classes 

# Step 2 will implement these
from source.envs.grasp_and_flip import GraspAndFlipEnv, GraspAndFlipEnvCfg

def launch(headless: bool, record_video: bool, return_env: bool = False):
    # 1) start Isaac Lab
    launcher = AppLauncher(headless=headless,
                           enable_cameras=record_video)
    app = launcher.app

    # 2) simulation setup
    sim = SimulationContext(physics_dt=1/60.0,
                            rendering_dt=1/30.0)

    # 3) load config
    cfg = yaml.safe_load(open("source/config.yaml"))

    # 4) create RL env
    env_cfg  = DirectRLEnvCfg(sim=sim, viewer=not headless)
    task_cfg = GraspAndFlipEnvCfg(**cfg["env"])
    env      = GraspAndFlipEnv(env_cfg, task_cfg)
    env.initialize()

    # 5) placeholder rollout
    obs = env.reset()
    for _ in range(cfg["train"]["total_steps"]):
        a = env.action_space.sample()
        obs, r, done, _ = env.step(a)
        if done:
            obs = env.reset()

    # At end of function:
    if return_env:
        return env
    else:
        # 6) shutdown
        app.close()
