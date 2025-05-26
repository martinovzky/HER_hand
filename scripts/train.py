#!/usr/bin/env python3

import argparse
import os
import yaml

from source.extension import launch
from source.agents.ddpg_her import make_ddpg_her_agent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="run without GUI")
    p.add_argument("--record",   action="store_true", help="enable cameras for recording")
    args = p.parse_args()

    cfg = yaml.safe_load(open("source/config.yaml"))

    # launch the sim+env and get back the env instance
    env = launch(
        headless=args.headless,
        record_video=args.record,
        return_env=True
    )

    # build agent
    train_cfg = {
        "batch_size":    cfg["train"].get("batch_size", 256),
        "learning_rate": cfg["train"].get("learning_rate", 1e-3),
    }
    her_cfg = {"k": cfg["her"].get("k", 4)}
    model = make_ddpg_her_agent(env, train_cfg, her_cfg)

    # train
    model.learn(total_timesteps=cfg["train"]["total_steps"])

    # ensure models directory exists
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # save policy; adjust this path if running on a VM
    model_path = os.path.join(model_dir, "ddpg_her")
    model.save(model_path)

    env.close()

if __name__ == "__main__":
    main()

