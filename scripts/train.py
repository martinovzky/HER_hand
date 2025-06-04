#!/usr/bin/env python3

import argparse
import os
import yaml
from pathlib import Path
from gymnasium.wrappers.record_video import RecordVideo

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

    # Get absolute path to config - this script is in HER_hand/scripts/train.py
    script_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/HER_hand/scripts
    project_root = os.path.dirname(script_dir)              # /path/to/HER_hand
    config_path = os.path.join(project_root, "source", "config.yaml")
    
    print(f"DEBUG: Script directory: {script_dir}")
    print(f"DEBUG: Project root: {project_root}")
    print(f"DEBUG: Config path: {config_path}")
    print(f"DEBUG: Config exists: {os.path.exists(config_path)}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    
    if not os.path.exists(config_path):
        # Try alternative paths
        alt_paths = [
            "source/config.yaml",  # relative from cwd
            "../source/config.yaml",  # relative from scripts dir
            os.path.join(os.getcwd(), "source", "config.yaml"),  # from cwd
        ]
        
        print("DEBUG: Trying alternative paths:")
        for alt_path in alt_paths:
            print(f"  {alt_path}: {os.path.exists(alt_path)}")
            if os.path.exists(alt_path):
                config_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Config file not found at any of these paths: {[config_path] + alt_paths}")
    
    print(f"DEBUG: Using config path: {config_path}")
    
    # Load and debug the config
    with open(config_path, 'r') as f:
        file_content = f.read()
        print(f"DEBUG: Raw file content:\n{file_content}")
    
    # Parse YAML
    cfg = yaml.safe_load(file_content)
    
    # Debug: Print what we actually loaded
    print(f"DEBUG: Config type: {type(cfg)}")
    print(f"DEBUG: Config content: {cfg}")
    print(f"DEBUG: Config keys: {list(cfg.keys()) if isinstance(cfg, dict) else 'Not a dict!'}")
    
    # Validate config structure
    if cfg is None:
        raise ValueError("Config file is empty or invalid")
    
    if isinstance(cfg, list):
        raise ValueError("Config file should contain a dictionary, not a list. Check your YAML structure.")
    
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file should contain a dictionary, got {type(cfg)}")

    # Enable cameras if video recording is requested (following Isaac Lab pattern)
    enable_cameras = args.video or args.record
    
    # launch the sim+env and get back the env instance
    env = launch(
        headless=args.headless,
        record_video=enable_cameras,
        return_env=True
    )

    # Check if env was created successfully
    if env is None:
        raise RuntimeError("Failed to create environment from launch()")

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

    # build agent, training, and HER configurations with safe access
    print(f"DEBUG: Accessing train config: {cfg.get('train', {})}")
    print(f"DEBUG: Accessing her config: {cfg.get('her', {})}")
    
    train_cfg = {
        "batch_size":    cfg.get("train", {}).get("batch_size", 256),
        "learning_rate": cfg.get("train", {}).get("learning_rate", 1e-3),
    }
    her_section = cfg.get("her", {}) if isinstance(cfg.get("her", {}), dict) else {}
    her_cfg = {
        "n_sampled_goal": her_section.get("n_sampled_goal", 4),
        "goal_selection_strategy": her_section.get("goal_selection_strategy", "future"),
        "online_sampling": her_section.get("online_sampling", True),
    }
    if "max_episode_length" in her_section:
        her_cfg["max_episode_length"] = her_section["max_episode_length"]
    
    print(f"DEBUG: Final train_cfg: {train_cfg}")
    print(f"DEBUG: Final her_cfg: {her_cfg}")

    # Ensure tensorboard log directory exists
    os.makedirs("logs", exist_ok=True)
    print("DEBUG: TensorBoard log directory created at 'logs/'")

    
    # Create agent with progress callback
    model, callback = make_ddpg_her_agent(env, train_cfg, her_cfg)

    total_steps = cfg.get("train", {}).get("total_steps", 100000)
    print(f"Starting training for {total_steps:,} steps...")
    print("Progress will be logged every 1000 steps")
    
    # train with callback for progress logging
    model.learn(
        total_timesteps=total_steps,
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



