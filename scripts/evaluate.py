#!/usr/bin/env python3

import yaml
from stable_baselines3 import DDPG

from source.extension import launch

def main():
    cfg = yaml.safe_load(open("source/config.yaml"))

    env = launch(headless=True, record_video=False, return_env=True)
    model = DDPG.load("models/ddpg_her", env=env)

    episodes = cfg.get("eval", {}).get("episodes", 100)
    successes = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        successes += int(reward == 1.0)

    rate = successes / episodes
    print(f"Evaluation: {successes}/{episodes} successes ({rate:.2%})")
    env.close()

if __name__ == "__main__":
    main()

