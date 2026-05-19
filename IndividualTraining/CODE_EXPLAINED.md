# Code Explanation: train.py (DQN) and train_ppo.py (PPO)

---

# Part 1: train.py — DQN from Scratch

## What this file does

Trains a DQN (Deep Q-Network) agent to play Pong from scratch on the Jetson Orin Nano.
The AI starts with zero knowledge and learns purely by playing the game repeatedly.

> Note: This version caused an OOM crash on Jetson due to the 5.2GB replay buffer.
> See DQN_PROBLEM_AND_PPO.md for details. Use train_ppo.py instead.

---

## Section 1: Imports

```python
import ale_py
import gymnasium
gymnasium.register_envs(ale_py)
```

- `ale_py` — The Atari Learning Environment. This is the actual Pong game engine. Without it, there is no game to play.
- `gymnasium` — The interface between the game and the AI. It standardizes how the AI sends actions and receives observations and rewards.
- `gymnasium.register_envs(ale_py)` — Tells gymnasium that Atari environments exist. Without this line, `PongNoFrameskip-v4` cannot be found and the code will crash.

```python
from stable_baselines3 import DQN
```

- Imports the DQN algorithm from Stable-Baselines3 (SB3). SB3 is a library of reliable, pre-implemented RL algorithms. We are using their DQN, not writing it from scratch.

```python
from stable_baselines3.common.env_util import make_atari_env
```

- A helper function that creates an Atari environment with all standard preprocessing already applied:
  - Converts game screen to grayscale
  - Resizes frames to 84×84 pixels
  - Clips rewards to -1, 0, or +1
  - Handles episode termination correctly

```python
from stable_baselines3.common.vec_env import VecFrameStack
```

- Stacks the last 4 game frames together before feeding them to the AI.
- A single frame shows where the ball is. Four stacked frames show which direction it is moving.

```python
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
```

- `BaseCallback` — Base class for writing custom callbacks. Our TrainingMonitor inherits from this.
- `CheckpointCallback` — Automatically saves the model every N steps.
- `EvalCallback` — Periodically runs the agent in a separate environment and saves the best model found.

```python
import warnings
warnings.filterwarnings("ignore")
```

- Suppresses irrelevant warning messages that SB3 and gymnasium print during training.

---

## Section 2: TrainingMonitor Class

```python
class TrainingMonitor(BaseCallback):
```

- Creates a custom callback by inheriting from `BaseCallback`.
- SB3 will call specific methods of this class automatically at certain points during training.

```python
def __init__(self, verbose=1):
    super().__init__(verbose)
    self.step_count = 0
```

- `__init__` runs once when the object is created. Initializes a step counter.
- `super().__init__(verbose)` — Runs the parent class (BaseCallback) initialization first.

```python
def _on_step(self) -> bool:
```

- SB3 calls this method automatically after every single environment step.
- Must return `True` to continue training, or `False` to stop early.

```python
if self.step_count == 1:
    print("PHASE A: Filling replay buffer...")
```

- On the very first step, prints an explanation that the AI is in pure exploration mode.
- No learning happens yet — the AI just takes random actions to fill the buffer.

```python
if self.step_count == 10_001:
    print("PHASE B: Learning has started!")
```

- At step 10,001 (right after `learning_starts=10_000`), the AI begins actual learning.
- Prints a message explaining that Q-value updates are now happening.

```python
if self.step_count % 10_000 == 0:
    exploration = self.model.exploration_rate
    buffer_size = self.model.replay_buffer.size()
    print(...)
```

- Every 10,000 steps, prints the current exploration rate and buffer size.
- `exploration_rate` starts high (mostly random) and decreases over time (more Q-value-based).
- `replay_buffer.size()` shows how many experiences are currently stored.

---

## Section 3: Environment Creation

```python
env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
```

- Creates 4 parallel Pong environments running simultaneously.
- `NoFrameskip` means every frame is shown to the AI (no frames skipped).
- `seed=0` fixes the random seed for reproducibility.
- `VecFrameStack(env, n_stack=4)` stacks 4 consecutive frames so the AI can perceive motion.

```python
eval_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)
```

- A separate environment used only for performance evaluation (not training).
- Different seed (42) so the AI is tested on slightly different conditions.

---

## Section 4: DQN Model

```python
model = DQN(
    "CnnPolicy",
    env,
    verbose=0,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=32,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    tensorboard_log="./logs/",
)
```

- `"CnnPolicy"` — Uses a Convolutional Neural Network to process raw pixel input.
- `learning_rate=1e-4` — How large each weight update is. `1e-4 = 0.0001`.
- `buffer_size=100_000` — Stores 100,000 past experiences. **This caused the 5.2GB OOM crash.**
- `learning_starts=10_000` — Does not begin learning until 10,000 experiences are collected.
- `batch_size=32` — Samples 32 random experiences from the buffer per update.
- `exploration_fraction=0.1` — Spends the first 10% of training on random exploration.
- `exploration_final_eps=0.01` — After exploration phase, keeps 1% randomness permanently.
- `tensorboard_log` — Saves training metrics so they can be visualized with TensorBoard.

### How DQN learns

```
Game screen → CNN → Q-values for each action
                    UP=0.7, STAY=0.2, DOWN=0.1
                              ↓
                    Pick highest Q-value → take action
                              ↓
                    Store (state, action, reward, next_state) in replay buffer
                              ↓
                    Sample 32 random past experiences
                              ↓
                    Update Q-values to better match actual rewards
```

---

## Section 5: Callbacks

```python
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="dqn_pong",
)
```

- Saves the model every 50,000 steps as `dqn_pong_50000_steps.zip`, etc.
- Allows resuming training from the last checkpoint if a crash occurs.

```python
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    eval_freq=50_000,
    n_eval_episodes=5,
    verbose=1,
)
```

- Every 50,000 steps, runs 5 test episodes in `eval_env`.
- If the average score is the best seen so far, saves the model to `./best_model/`.
- `verbose=1` prints the evaluation results.

---

## Section 6: Training and Saving

```python
model.learn(
    total_timesteps=1_000_000,
    callback=[monitor_callback, checkpoint_callback, eval_callback],
)
```

- The single line that starts all training.
- Runs for 1,000,000 steps total.
- All three callbacks run simultaneously during training.

```python
model.save("dqn_pong_final")
```

- Saves the fully trained model as `dqn_pong_final.zip` when training completes.

---
---

# Part 2: train_ppo.py — PPO from Scratch

## What this file does

Trains a PPO (Proximal Policy Optimization) agent to play Pong from scratch.
PPO was chosen over DQN because it does not use a replay buffer, keeping memory usage around 200MB instead of 5.2GB — safe for the Jetson Orin Nano.

---

## Section 1: Imports

Same as train.py except:

```python
from stable_baselines3 import PPO
```

- Imports PPO instead of DQN. Everything else (gymnasium, ale_py, callbacks) is identical because the game environment setup is the same regardless of the algorithm.

---

## Section 2: PPOTrainingMonitor Class

```python
class PPOTrainingMonitor(BaseCallback):
    def __init__(self, n_steps, n_envs, verbose=1):
        super().__init__(verbose)
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.rollout_size = n_steps * n_envs  # 128 * 8 = 1,024
        self.rollout_count = 0
        self.step_count = 0
```

- Takes `n_steps` and `n_envs` as arguments to calculate rollout size.
- `rollout_size = 1,024` — experiences collected before each policy update.
- Tracks both step count and rollout count separately.

```python
def _on_step(self) -> bool:
    self.step_count += 1
    if self.step_count == 1:
        print("PHASE A: Collecting first rollout")
    return True
```

- Called every step. On step 1, explains that the first rollout is being collected.
- PPO does not have a separate "fill buffer" phase like DQN — it starts collecting and then immediately learns.

```python
def _on_rollout_end(self) -> None:
    self.rollout_count += 1
    if self.rollout_count == 1:
        print("PHASE B: First policy update complete!")
    if self.rollout_count % 10 == 0:
        entropy = self.locals.get("entropy_loss", None)
        print(...)
```

- `_on_rollout_end` is called by SB3 after each complete rollout (every 1,024 steps).
- Unlike DQN's `_on_step`, this fires once per training cycle, not once per step.
- `self.locals.get("entropy_loss")` reads the entropy value from SB3's internal state.
- Entropy measures how diverse the AI's actions are. High entropy = still exploring. Low entropy = mostly decided on one strategy.

```python
print(f"  Approx memory usage : ~{self.rollout_size * 4 * 84 * 84 * 4 / 1e6:.0f}MB")
```

- Calculates rollout buffer memory: `1024 experiences × 4 frames × 84×84 pixels × 4 bytes ≈ 115MB`

---

## Section 3: Environment Creation

```python
env = make_atari_env("PongNoFrameskip-v4", n_envs=8, seed=0)
env = VecFrameStack(env, n_stack=4)
```

- Uses 8 parallel environments instead of DQN's 4.
- PPO benefits more from parallel environments because it needs diverse experiences within each rollout.
- More environments = more varied experiences collected per cycle = better learning.

---

## Section 4: PPO Model

```python
model = PPO(
    "CnnPolicy",
    env,
    verbose=0,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    learning_rate=2.5e-4,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    gae_lambda=0.95,
    gamma=0.99,
    tensorboard_log="./logs_ppo/",
)
```

### Parameter-by-parameter explanation

**`"CnnPolicy"`**  
Uses a CNN to read raw pixel input, same as DQN.

**`n_steps=128`**  
Each environment plays 128 steps before a policy update happens.  
Total experiences per update: `128 × 8 environments = 1,024`.  
This is the rollout buffer — it is created, used, and deleted every cycle.

**`batch_size=256`**  
The 1,024 experiences are split into mini-batches of 256 for gradient updates.  
`1024 ÷ 256 = 4 mini-batches per epoch`.

**`n_epochs=4`**  
The same 1,024 experiences are reused for 4 full passes.  
DQN uses each experience once. PPO reuses them 4 times before discarding.

**`learning_rate=2.5e-4`**  
`0.00025`. Controls how large each weight update is.  
Slightly higher than DQN's `1e-4` because PPO's clipping makes larger rates safer.

**`clip_range=0.2`**  
The "Proximal" part of PPO. Limits how much the policy can change per update to ±20%.  
Prevents a single good experience from pushing the policy to an extreme.

**`ent_coef=0.01`**  
Entropy coefficient. Adds a small bonus reward for taking diverse actions.  
Prevents the AI from locking onto one action too early in training.  
Too high = AI never converges. Too low = AI stops exploring too soon.

**`vf_coef=0.5`**  
Value function coefficient. Controls how much the Critic's loss contributes to the total loss.  
The total loss = Actor loss + 0.5 × Critic loss + entropy bonus.

**`gae_lambda=0.95`**  
Controls how far into the future the Advantage estimate looks.  
`λ=1.0` uses all future rewards (high variance). `λ=0.0` uses only immediate reward (high bias).  
`λ=0.95` is the standard balance point.

**`gamma=0.99`**  
Discount factor for future rewards.  
A reward 10 steps in the future is worth `0.99^10 = 0.90` of an immediate reward.  
High gamma = AI cares about long-term outcomes.

**`tensorboard_log="./logs_ppo/"`**  
Saves training curves to a separate folder from DQN logs.

### How PPO learns (one full cycle)

```
1. Play 128 steps × 8 environments = 1,024 experiences collected
2. Critic scores each experience → compute Advantage (better or worse than expected?)
3. Actor updates: increase probability of high-advantage actions
   → clipped to ±20% maximum change
4. Repeat step 3 for 4 epochs using the same 1,024 experiences
5. Discard all 1,024 experiences
6. Go back to step 1
```

---

## Section 5: Callbacks

```python
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints_ppo/",
    name_prefix="ppo_pong",
)
```

- Saves `ppo_pong_50000_steps.zip`, `ppo_pong_100000_steps.zip`, etc.
- Separate folder from DQN checkpoints to avoid mixing models.

```python
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model_ppo/",
    eval_freq=50_000,
    n_eval_episodes=5,
    verbose=1,
)
```

- Same structure as DQN. Every 50,000 steps, runs 5 test games.
- Saves the best-performing model seen throughout training.

---

## Section 6: Training and Saving

```python
model.learn(
    total_timesteps=1_000_000,
    callback=[monitor_callback, checkpoint_callback, eval_callback],
)
```

- The single line where all learning happens.
- Everything else in the file is setup, monitoring, or saving.

```python
model.save("ppo_pong_final")
```

- Saves the final model as `ppo_pong_final.zip`.

---

## File Summary

| File | Algorithm | Memory | Status |
|------|-----------|--------|--------|
| `train.py` | DQN | ~5.2GB | Caused OOM reboot on Jetson |
| `train_ppo.py` | PPO | ~200MB | Safe for Jetson Orin Nano |
