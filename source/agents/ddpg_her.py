import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.her.goal_env_wrapper import HERGoalEnvWrapper

class ProgressCallback(BaseCallback):
    """Enhanced callback to log comprehensive training progress"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.success_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_rewards = []  # Last 100 episodes
        self.recent_lengths = []  # Last 100 episodes
        
    def _on_step(self) -> bool:
        # Log every 1000 steps with comprehensive metrics
        if self.num_timesteps % 1000 == 0:
            print(f"\n{'='*60}")
            print(f"Training Step: {self.num_timesteps:,} / 1,000,000 ({self.num_timesteps/10000:.1f}%)")
            
            # Show learning rate if available
            if hasattr(self.model, 'learning_rate'):
                lr = self.model.learning_rate
                if callable(lr):
                    lr = lr(1.0)  # Get current learning rate
                print(f"Learning Rate: {lr:.6f}")
            
            # Show DDPG losses if available
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logger_data = self.model.logger.name_to_value
                if 'train/actor_loss' in logger_data:
                    print(f"Actor Loss:    {logger_data['train/actor_loss']:.6f}")
                if 'train/critic_loss' in logger_data:
                    print(f"Critic Loss:   {logger_data['train/critic_loss']:.6f}")
        
        # Track episodes and comprehensive metrics
        if 'episode' in self.locals.get('infos', [{}])[0]:
            self.episode_count += 1
            episode_info = self.locals['infos'][0]['episode']
            
            reward = episode_info['r']
            length = episode_info['l']
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            
            # Keep recent history (last 100 episodes)
            self.recent_rewards.append(reward)
            self.recent_lengths.append(length)
            if len(self.recent_rewards) > 100:
                self.recent_rewards.pop(0)
                self.recent_lengths.pop(0)
            
            # Count successes
            if reward > 0.5:  # Success if reward > 0.5
                self.success_count += 1
                
            # Log detailed metrics every 100 episodes
            if self.episode_count % 100 == 0:
                success_rate = self.success_count / self.episode_count
                
                # Recent performance (last 100 episodes)
                recent_success_rate = sum(1 for r in self.recent_rewards if r > 0.5) / len(self.recent_rewards)
                recent_avg_reward = np.mean(self.recent_rewards)
                recent_avg_length = np.mean(self.recent_lengths)
                
                # Overall performance
                overall_avg_reward = np.mean(self.episode_rewards)
                overall_avg_length = np.mean(self.episode_lengths)
                
                print(f"\n{'='*60}")
                print(f"EPISODE MILESTONE: {self.episode_count}")
                print(f"{'='*60}")
                print(f"Overall Success Rate:     {success_rate:.2%}")
                print(f"Recent Success Rate:      {recent_success_rate:.2%} (last 100 episodes)")
                print(f"")
                print(f"Reward Metrics:")
                print(f"  Recent Avg Reward:      {recent_avg_reward:.3f}")
                print(f"  Overall Avg Reward:     {overall_avg_reward:.3f}")
                print(f"")
                print(f"Episode Length Metrics:")
                print(f"  Recent Avg Length:      {recent_avg_length:.1f} steps")
                print(f"  Overall Avg Length:     {overall_avg_length:.1f} steps")
                print(f"")
                print(f"Training Progress:")
                print(f"  Total Episodes:         {self.episode_count}")
                print(f"  Total Successes:        {self.success_count}")
                print(f"  Training Steps:         {self.num_timesteps:,}")
                
                # Performance trend
                if self.episode_count >= 200:
                    prev_100_success = sum(1 for r in self.episode_rewards[-200:-100] if r > 0.5) / 100
                    current_100_success = recent_success_rate
                    trend = "↗️ IMPROVING" if current_100_success > prev_100_success else "↘️ DECLINING" if current_100_success < prev_100_success else "→ STABLE"
                    print(f"  Performance Trend:      {trend}")
                
                print(f"{'='*60}")
        
        return True

def make_ddpg_her_agent(env, train_cfg: dict, her_cfg: dict):
    """
    Wrap `env` with HER and create a DDPG model.

    Args:
        env: a gym.GoalEnv-compliant environment (e.g., DirectRLEnv)
        train_cfg: dict with keys:
            - batch_size: int
            - learning_rate: float
        her_cfg: dict with key:
            - k: int (number of future goals to sample per step)
    Returns:
        model: a Stable-Baselines3 DDPG model wrapped for HER
        callback: progress callback for training
    """
    # Wrap environment with HER
    her_env = HERGoalEnvWrapper(
        env,
        n_sampled_goal=her_cfg["k"],
        goal_selection_strategy="future",
        online_sampling=True
    )

    # Create DDPG model with TensorBoard logging
    model = DDPG(
        policy="MlpPolicy",
        env=her_env,
        batch_size=train_cfg.get("batch_size", 256),
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        tensorboard_log="./logs/",
        verbose=1
    )
    
    # Create enhanced progress callback
    callback = ProgressCallback()
    
    return model, callback
