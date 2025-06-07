# HER_Hand  

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
- Uses quaternion angular distance calculation

### 4. Agent (`source/agents/ddpg_her.py`)
- **make_ddpg_her_agent()**: Wraps environment with HERGoalEnvWrapper
- Uses "future" goal selection strategy with k=4 relabels per transition
- Creates DDPG model with specified hyperparameters


## Notes 

All code follows the new Isaac Sim 4.5 API structure:
- **No deprecated `omni.isaac.*` imports**
- **Uses Isaac Lab 2.0 APIs**: `isaaclab.envs`, `isaaclab.sim.spawners`
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
