import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import BaseCallback

class ProgressCallback(BaseCallback):
    """Enhanced callback to log comprehensive training progress"""
    def __init__(self, verbose: int = 0):
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
        infos = self.locals.get('infos', [{}])
        if len(infos) > 0 and 'episode' in infos[0]:
            self.episode_count += 1
            episode_info = infos[0]['episode']

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

            # Count successes (assuming reward > 0.5 means success)
            if reward > 0.5:
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
                print()
                print(f"Reward Metrics:")
                print(f"  Recent Avg Reward:      {recent_avg_reward:.3f}")
                print(f"  Overall Avg Reward:     {overall_avg_reward:.3f}")
                print()
                print(f"Episode Length Metrics:")
                print(f"  Recent Avg Length:      {recent_avg_length:.1f} steps")
                print(f"  Overall Avg Length:     {overall_avg_length:.1f} steps")
                print()
                print(f"Training Progress:")
                print(f"  Total Episodes:         {self.episode_count}")
                print(f"  Total Successes:        {self.success_count}")
                print(f"  Training Steps:         {self.num_timesteps:,}")

                # Performance trend (compare last 100 vs. previous 100 episodes)
                if self.episode_count >= 200:
                    prev_100_success = sum(
                        1 for r in self.episode_rewards[-200:-100] if r > 0.5
                    ) / 100
                    current_100_success = recent_success_rate
                    if current_100_success > prev_100_success:
                        trend = "↗️ IMPROVING"
                    elif current_100_success < prev_100_success:
                        trend = "↘️ DECLINING"
                    else:
                        trend = "→ STABLE"
                    print(f"  Performance Trend:      {trend}")

                print(f"{'='*60}")

        return True


def make_ddpg_her_agent(env, train_cfg: dict, her_cfg: dict):
    """
    Create a DDPG model with HER using SB3's HerReplayBuffer.

    Args:
        env: a goal-conditioned environment. 
          It should return dict observations with keys ``'observation'``, ``'achieved_goal'`` and
             ``'desired_goal'`` and expose ``compute_reward()`` for HER.
        train_cfg: dict with keys:
            - batch_size: int
            - learning_rate: float
        her_cfg: dict with keys:
            - n_sampled_goal: int  (number of future goals to sample per step)
            - goal_selection_strategy: str or GoalSelectionStrategy enum 
                                       (e.g., "future" or GoalSelectionStrategy.FUTURE)
            - copy_info_dict: bool (optional)

    Returns:
        model: a Stable-Baselines3 DDPG model configured with HerReplayBuffer
        callback: an instance of ProgressCallback for detailed logging
    """

    # Build the replay_buffer_kwargs exactly as required by HerReplayBuffer
    replay_buffer_kwargs = {
        "n_sampled_goal": her_cfg.get("n_sampled_goal", 4),
        "goal_selection_strategy": her_cfg.get(
            "goal_selection_strategy", GoalSelectionStrategy.FUTURE
        ),
        "copy_info_dict": her_cfg.get("copy_info_dict", False),
    }

    # Create the DDPG model, passing in the HER replay buffer
    model = DDPG(
        policy="MultiInputPolicy",            # Required for Dict obs: {observation, achieved_goal, desired_goal}
        env=env,                               # Pass the GoalEnv directly
        replay_buffer_class=HerReplayBuffer,   # Hook in HER's replay buffer
        replay_buffer_kwargs=replay_buffer_kwargs,
        batch_size=train_cfg.get("batch_size", 256),
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        learning_starts=2000,                  # Wait for 2000 steps before starting training
        train_freq=1,                          # Train every step after learning_starts
        gradient_steps=1,                      # Number of gradient steps per training
        tensorboard_log="./logs/",
        verbose=1,
    )

    # Return both model and your ProgressCallback
    return model, ProgressCallback()

