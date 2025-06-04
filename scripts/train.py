#!/usr/bin/env python3

import argparse
import os
import yaml
from pathlib import Path
from gymnasium.wrappers import RecordVideo

from source.extension import launch
from source.agents.ddpg_her import make_ddpg_her_agent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="run without GUI")
    p.add_argument("--record", action="store_true", help="enable cameras for recording (deprecated, use --video)")
    
    # Isaac Lab video recording arguments (following their exact implementation)
    p.add_argument("--video", action="store_true", help="enable video recording during training")
    p.add_argument("--video_length", type=int, default=900, help="length of each recorded video (in steps)")
    p.add_argument("--video_interval", type=int, default=18000, help="interval between each video recording (in steps)")
    
    args = p.parse_args()

    cfg = yaml.safe_load(open("source/config.yaml"))

    # Enable cameras if video recording is requested (following Isaac Lab pattern)
    enable_cameras = args.video or args.record
    
    # launch the sim+env and get back the env instance
    env = launch(
        headless=args.headless,
        record_video=enable_cameras,
        return_env=True
    )

    # Apply video recording wrapper if requested (Isaac Lab's exact approach)
    if args.video:
        # Create videos directory on Desktop
        desktop_path = Path.home() / "Desktop"
        video_dir = desktop_path / "HER_hand_videos"
        video_dir.mkdir(exist_ok=True)
        
        print(f"[INFO] Video recording enabled:")
        print(f"  Video length: {args.video_length} steps (~30 seconds)")
        print(f"  Video interval: {args.video_interval} steps (~10 minutes)")
        print(f"  Videos saved to: {video_dir}")
        
        # Wrap with RecordVideo following Isaac Lab's pattern
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: episode_id % max(1, args.video_interval // 50) == 0,
            video_length=args.video_length,
            name_prefix="train"
        )

    # build agent, training, and HER configurations
    train_cfg = {
        "batch_size":    cfg["train"].get("batch_size", 256),
        "learning_rate": cfg["train"].get("learning_rate", 1e-3),
    }
    
    her_section = cfg.get("her", {}) if isinstance(cfg.get("her", {}), dict) else {}
    her_cfg = {
    "n_sampled_goal": her_section.get("n_sampled_goal", 4),
    "goal_selection_strategy": her_section.get("goal_selection_strategy", "future"),
    "online_sampling": her_section.get("online_sampling", True),
    }
    
if "max_episode_length" in her_section:
    her_cfg["max_episode_length"] = her_section["max_episode_length"]

    
    # Create agent with progress callback
    model, callback = make_ddpg_her_agent(env, train_cfg, her_cfg)

    print(f"Starting training for {cfg['train']['total_steps']:,} steps...")
    print("Progress will be logged every 1000 steps")
    
    # train with callback for progress logging
    model.learn(
        total_timesteps=cfg["train"]["total_steps"],
        callback=callback
    )

    # ensure models directory exists
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # save policy
    model_path = os.path.join(model_dir, "ddpg_her")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    env.close()

if __name__ == "__main__":
    main()

