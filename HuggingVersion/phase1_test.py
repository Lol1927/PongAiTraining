import ale_py
import gymnasium
gymnasium.register_envs(ale_py)

from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import warnings
warnings.filterwarnings("ignore")

print("모델 다운로드 중...")
checkpoint = load_from_hub(
    repo_id="sb3/dqn-PongNoFrameskip-v4",
    filename="dqn-PongNoFrameskip-v4.zip",
)

print("환경 초기화 중...")
env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
env = VecFrameStack(env, n_stack=4)

print("모델 로딩 중...")
custom_objects = {
    "learning_rate": 1e-4,
    "lr_schedule": lambda _: 1e-4,
    "exploration_schedule": lambda _: 0.01,
    "optimize_memory_usage": False,
}
model = DQN.load(checkpoint, env=env, custom_objects=custom_objects)

print("AI 플레이 시작! (Ctrl+C로 종료)")
episode = 0
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done[0]:
        episode += 1
        score = info[0].get("episode", {}).get("r", "?")
        print(f"Episode {episode} 완료 - 점수: {score}", flush=True)
        obs = env.reset()
