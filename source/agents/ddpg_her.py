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
        # The default SB3 logging (every 1000 steps) will continue as normal
        # We don't need to add any custom step-based logging here
        
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

            # Log detailed metrics every 2500 episodes 
            if self.episode_count % 2500 == 0:
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
                if len(self.episode_rewards) >= 200:
                    last_100_avg = np.mean(self.episode_rewards[-100:])
                    prev_100_avg = np.mean(self.episode_rewards[-200:-100])
                    if last_100_avg > prev_100_avg + 0.01:
                        trend = "↗ IMPROVING"
                    elif last_100_avg < prev_100_avg - 0.01:
                        trend = "↘ DECLINING"
                    else:
                        trend = "→ STABLE"
                    print(f"  Performance Trend:      {trend}")
                else:
                    print(f"  Performance Trend:      → STABLE")
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
        learning_starts=1000,                  # Wait for 1000 steps before starting training
        train_freq=1,                          # Train every step after learning_starts
        gradient_steps=1,                      # Number of gradient steps per training
        tensorboard_log="./logs/",
        verbose=1,
    )

    # Return both model and your ProgressCallback
    return model, ProgressCallback()
