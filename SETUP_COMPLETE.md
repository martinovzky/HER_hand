# HER Hand Project - Setup Complete ✅

## Summary

Successfully implemented a complete "grasp & flip" HER pipeline using Isaac Sim 4.5 & Isaac Lab 2.0 APIs.

### Task Description
- **Objective**: Pick up a cube with a Shadow Hand, then rotate it 180° ("flip")
- **Reward**: Sparse binary reward (1.0 for success, 0.0 otherwise)
- **Learning**: Hindsight Experience Replay (HER) with DDPG

## Key Components Implemented

### 1. Environment (`source/envs/grasp_and_flip.py`)
- **GraspAndFlipEnv**: Subclass of DirectRLEnv
- **build_scene()**: Spawns table, cube, and Shadow Hand using USD assets
- **reset()**: Closes hand to grasp, computes flipped goal orientation
- **get_observations()**: Goal-conditioned observations
- **compute_reward()**: Binary reward using position and orientation tolerances

### 2. USD Assets (`assets/`)
- **table_at_hand_height.usd**: 60cm x 60cm table surface at 0.3m height
- **cube.usd**: 5cm red cube with physics properties (mass, friction, collision)

### 3. Reward Function (`source/utils/reward.py`)
- **sparse_grasp_flip_reward()**: Returns 1.0 only if both position & orientation errors are within tolerances
- Uses robust quaternion angular distance calculation

### 4. Agent (`source/agents/ddpg_her.py`)
- **make_ddpg_her_agent()**: Wraps environment with HERGoalEnvWrapper
- Uses "future" goal selection strategy with k=4 relabels per transition
- Creates DDPG model with specified hyperparameters

### 5. Launcher (`source/extension.py`)
- **launch()**: Starts Isaac Lab via AppLauncher
- Creates SimulationContext with proper physics/rendering timesteps
- Instantiates environment with DirectRLEnvCfg
- Supports headless mode and video recording

### 6. Training Scripts
- **scripts/train.py**: Complete training pipeline with CLI arguments
- **scripts/evaluate.py**: Evaluation script for trained models

## Object Positioning Strategy

### Optimal Placement for Reachability
```
Hand Position:    (0.0, 0.0, 0.45)  # 15cm above table
Cube Position:    (0.0, -0.1, 0.325) # 10cm in front of hand, on table
Table Surface:    Z = 0.3m
```

### Why This Works
- **Cube is 10cm in front of hand**: Within Shadow Hand's reach envelope
- **Hand is 15cm above table**: Allows proper grasp approach angle
- **Table height (30cm)**: Standard height for robotic manipulation
- **Cube size (5cm)**: Optimal for Shadow Hand finger span

## Configuration (`source/config.yaml`)

```yaml
env:
  table_height: 0.3    # Table surface at 30cm
  flip_axis: [0, 1, 0] # Flip around Y-axis
  flip_angle: 3.1416   # 180° rotation
  pos_tol: 0.02        # 2cm position tolerance
  ori_tol: 0.1         # ~6° orientation tolerance
  grasp_angle: 0.8     # Hand closure target

train:
  total_steps: 1000000 # 1M training steps
  batch_size: 256
  learning_rate: 0.001

her:
  n_sampled_goal: 4            # number of future goals sampled
  goal_selection_strategy: future
  online_sampling: true
```

## Isaac Sim 4.5 Compatibility ✅

All code follows the new Isaac Sim 4.5 API structure:
- **No deprecated `omni.isaac.*` imports**
- **Uses Isaac Lab 2.0 APIs**: `isaaclab.envs`, `isaaclab.sim.spawners`
- **Modern spawning**: `spawn_usd` with `UsdFileCfg`
- **DirectRLEnv**: Latest RL environment base class

## Usage

### Training
```bash
# Headless training
python scripts/train.py --headless

# With GUI for debugging
python scripts/train.py

# With video recording
python scripts/train.py --record
```

### Headless recording with `isaac.bat`
Use the Isaac Sim batch launcher to run headless with video:

```cmd
isaac-sim.bat --no-window --/app/renderer/enable_recording=true -- \ 
  python scripts/train.py --headless --video


```

### Evaluation
```bash
python scripts/evaluate.py
```

## Dependencies Required

```bash
pip install stable-baselines3 sb3-contrib pyyaml numpy tensorboard```

Plus Isaac Sim 4.5 and Isaac Lab 2.0 installation

TensorBoard logs are saved to the `logs/` directory. Launch with:

```bash
tensorboard --logdir logs
```

## Expected Learning Behavior

1. **Early Training**: Random actions, low success rate
2. **HER Effect**: Learns to manipulate cube through hindsight relabeling
3. **Convergence**: Achieves consistent grasping and 180° flipping

The sparse reward makes this a challenging task that benefits greatly from HER's sample efficiency improvements.
