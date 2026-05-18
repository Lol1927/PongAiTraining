import ale_py
import gymnasium
gymnasium.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import warnings
warnings.filterwarnings("ignore")


class TrainingMonitor(BaseCallback):
    """Prints a detailed summary of what the AI is doing at each training stage."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1

        # ── Phase A: Filling the replay buffer (pure exploration) ──
        if self.step_count == 1:
            print("=" * 60)
            print("PHASE A: Filling replay buffer (random actions only)")
            print(f"  The AI will take random actions for the first 10,000 steps.")
            print(f"  Every experience is stored in memory (replay buffer).")
            print(f"  No learning yet — just collecting experiences.")
            print("=" * 60)

        # ── Phase B: Learning begins ──
        if self.step_count == 10_001:
            print()
            print("=" * 60)
            print("PHASE B: Learning has started!")
            print(f"  Replay buffer now has enough experiences.")
            print(f"  Every step the AI now:")
            print(f"    1. Picks 32 random past experiences from memory")
            print(f"    2. Checks what reward each experience gave")
            print(f"    3. Adjusts Q-values to match better outcomes")
            print(f"  Exploration rate: {self.model.exploration_rate:.2f} (still mostly random)")
            print("=" * 60)

        # ── Print detailed info every 10,000 steps ──
        if self.step_count % 10_000 == 0:
            exploration = self.model.exploration_rate
            buffer_size = self.model.replay_buffer.size()

            print()
            print(f"─── Step {self.step_count:,} ───────────────────────────────")
            print(f"  Replay buffer     : {buffer_size:,} experiences stored")
            print(f"  Exploration rate  : {exploration:.3f}  ({exploration*100:.1f}% random actions)")
            print(f"  Remaining rate    : {1-exploration:.3f}  ({(1-exploration)*100:.1f}% Q-value based actions)")

            if exploration > 0.5:
                print(f"  Status: Still exploring — AI mostly guessing")
            elif exploration > 0.1:
                print(f"  Status: Transitioning — AI starting to use Q-values")
            else:
                print(f"  Status: Exploiting — AI mostly using learned Q-values")
            print()

        return True


# ── 1. Create environments ────────────────────────────
# Training: 4 parallel environments for faster learning
# No render_mode → no display → faster training
env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

# Evaluation: 1 environment to measure performance during training
eval_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)

# ── 2. Create DQN model ───────────────────────────────
model = DQN(
    "CnnPolicy",              # CNN reads raw pixel input from the screen
    env,
    verbose=0,                # We handle our own printing via TrainingMonitor
    learning_rate=1e-4,       # How fast Q-values are updated
    buffer_size=100_000,      # How many past experiences to store in memory
    learning_starts=10_000,   # Start learning only after this many steps
    batch_size=32,            # Number of experiences to learn from at once
    exploration_fraction=0.1, # Explore randomly for the first 10% of training
    exploration_final_eps=0.01,# Minimum exploration rate: 1%
    tensorboard_log="./logs/",
)

# ── 3. Set up callbacks ───────────────────────────────
monitor_callback = TrainingMonitor()

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="dqn_pong",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    eval_freq=50_000,
    n_eval_episodes=5,
    verbose=1,
)

# ── 4. Start training ─────────────────────────────────
print("=" * 60)
print("DQN Training — Starting from scratch")
print("=" * 60)
print(f"  Total steps     : 1,000,000")
print(f"  Environments    : 4 running in parallel")
print(f"  Learning starts : after 10,000 steps")
print(f"  Batch size      : 32 experiences per update")
print(f"  Save checkpoint : every 50,000 steps")
print()

model.learn(
    total_timesteps=1_000_000,
    callback=[monitor_callback, checkpoint_callback, eval_callback],
)

# ── 5. Save final model ───────────────────────────────
model.save("dqn_pong_final")
print()
print("=" * 60)
print("Training complete! Model saved: dqn_pong_final.zip")
print("To test the trained model, run:  python3 test.py")
print("=" * 60)
