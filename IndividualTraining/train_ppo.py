import ale_py
import gymnasium
gymnasium.register_envs(ale_py)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import warnings
warnings.filterwarnings("ignore")


class PPOTrainingMonitor(BaseCallback):
    """Prints a detailed summary of what the AI is doing at each training stage."""

    def __init__(self, n_steps, n_envs, verbose=1):
        super().__init__(verbose)
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.rollout_size = n_steps * n_envs  # experiences collected per cycle
        self.rollout_count = 0
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1

        if self.step_count == 1:
            print("=" * 60)
            print("PHASE A: Collecting first rollout")
            print(f"  Playing {self.n_steps} steps across {self.n_envs} environments")
            print(f"  Collecting {self.rollout_size:,} experiences before first update")
            print(f"  No replay buffer — experiences are discarded after each update")
            print("=" * 60)

        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1

        if self.rollout_count == 1:
            print()
            print("=" * 60)
            print("PHASE B: First policy update complete!")
            print(f"  Updated using {self.rollout_size:,} experiences")
            print(f"  All experiences discarded — fresh collection starts now")
            print("=" * 60)

        if self.rollout_count % 10 == 0:
            total_steps = self.rollout_count * self.rollout_size
            entropy = self.locals.get("entropy_loss", None)

            print()
            print(f"--- Rollout {self.rollout_count:,} (total steps: {total_steps:,}) ---")
            print(f"  Completed rollouts    : {self.rollout_count:,}")
            print(f"  Total steps processed : {total_steps:,}")
            if entropy is not None:
                print(f"  Exploration entropy   : {abs(entropy):.4f}  (higher = more diverse actions)")
            print(f"  Approx memory usage   : ~{self.rollout_size * 4 * 84 * 84 * 4 / 1e6:.0f}MB (rollout buffer only)")
            print()


# ── 1. Create environments ────────────────────────────
# n_envs=8: 8 parallel environments collect diverse experiences simultaneously
env = make_atari_env("PongNoFrameskip-v4", n_envs=8, seed=0)
env = VecFrameStack(env, n_stack=4)  # stack 4 frames so the agent sees ball movement

eval_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)

# ── 2. Create PPO model ───────────────────────────────
#
# n_steps=128
#   Play 128 steps per environment before each update.
#   Total experiences per update: 128 * 8 = 1,024.
#
# batch_size=256
#   Split the 1,024 experiences into mini-batches of 256 for gradient updates.
#
# n_epochs=4
#   Reuse the same 1,024 experiences for 4 update passes (unlike DQN which uses each once).
#
# clip_range=0.2
#   The "Proximal" part: limit policy change to ±20% of old probabilities per update.
#   Prevents catastrophically large updates that break the policy.
#
# ent_coef=0.01
#   Entropy bonus: reward the agent for taking diverse actions.
#   Prevents premature convergence to a single repeated action.
#
# vf_coef=0.5
#   Weight of the Critic (value function) loss relative to the Actor loss.
#
# gae_lambda=0.95
#   Controls how far into the future the advantage estimate looks.
#   0.95 means future rewards are discounted by 5% per step.
#
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

# ── 3. Set up callbacks ───────────────────────────────
monitor_callback = PPOTrainingMonitor(n_steps=128, n_envs=8)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints_ppo/",
    name_prefix="ppo_pong",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model_ppo/",
    eval_freq=50_000,
    n_eval_episodes=5,
    verbose=1,
)

# ── 4. Start training ─────────────────────────────────
print("=" * 60)
print("PPO Training — Starting from scratch")
print("=" * 60)
print(f"  Total timesteps   : 1,000,000")
print(f"  Environments      : 8 running in parallel")
print(f"  Rollout size      : 128 steps x 8 envs = 1,024 per cycle")
print(f"  Total rollouts    : ~{1_000_000 // (128 * 8):,}")
print(f"  Memory usage      : ~200MB  (vs DQN ~5.2GB)")
print(f"  Save checkpoint   : every 50,000 steps")
print()
print("  [How PPO differs from DQN]")
print("  DQN : stores 100,000 past experiences -> samples randomly")
print("  PPO : collects 1,024 experiences -> updates -> discards -> repeat")
print()

model.learn(
    total_timesteps=1_000_000,
    callback=[monitor_callback, checkpoint_callback, eval_callback],
)

# ── 5. Save final model ───────────────────────────────
model.save("ppo_pong_final")
print()
print("=" * 60)
print("Training complete! Model saved as: ppo_pong_final.zip")
print("To test the trained model, run:  python3 test_ppo.py")
print("=" * 60)
