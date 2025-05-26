from sb3_contrib import HERGoalEnvWrapper
from stable_baselines3 import DDPG

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
    """
    # Wrap environment with HER, grabs goals from future states
    
    her_env = HERGoalEnvWrapper(
        env,
        n_sampled_goal=her_cfg["k"], #k is in config.yaml
        goal_selection_strategy="future",
        online_sampling=True
    )

    # Create DDPG model, creates the replay buffer, where 
    model = DDPG(
        policy="MlpPolicy",
        env=her_env,
        batch_size=train_cfg.get("batch_size", 256),
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        tensorboard_log="./logs/",
        verbose=1
    )
    
    return model
