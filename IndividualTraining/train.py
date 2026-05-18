import ale_py
import gymnasium
gymnasium.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import warnings
warnings.filterwarnings("ignore")

# ── 1. Create environments ────────────────────────────
# Training: 4 parallel environments for faster learning
# No render_mode → no display → faster training
env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

# Evaluation: 1 environment to measure performance during training
eval_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)

# ── 2. Create DQN model ───────────────────────────────
# Unlike HuggingFace, this model starts from scratch with no prior knowledge
model = DQN(
    "CnnPolicy",              # CNN reads raw pixel input from the screen
    env,
    verbose=1,                # Print training progress to terminal
    learning_rate=1e-4,       # How fast Q-values are updated
    buffer_size=100_000,      # How many past experiences to store in memory
    learning_starts=10_000,   # Start learning only after this many steps
    batch_size=32,            # Number of experiences to learn from at once
    exploration_fraction=0.1, # Explore randomly for the first 10% of training
    exploration_final_eps=0.01, # Minimum exploration rate: 1%
    tensorboard_log="./logs/",  # Save training graph data here
)

# ── 3. Set up callbacks ───────────────────────────────
# Save model automatically every 50,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="dqn_pong",
)

# Measure performance every 50,000 steps and save the best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    eval_freq=50_000,
    n_eval_episodes=5,
    verbose=1,
)

# ── 4. Start training ─────────────────────────────────
print("Training started! Press Ctrl+C to stop.")
print("To monitor training progress, run in another terminal:")
print("  tensorboard --logdir ./logs/")
print()

model.learn(
    total_timesteps=1_000_000,
    callback=[checkpoint_callback, eval_callback],
)

# ── 5. Save final model ───────────────────────────────
model.save("dqn_pong_final")
print("Training complete! Model saved: dqn_pong_final.zip")
