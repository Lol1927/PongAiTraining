# Code Explanation: Phase 1

A detailed explanation of every library and code line used in `phase1_test.py`.

---

## Libraries Used

### Stable-Baselines3 (SB3)

SB3 stands for **Stable-Baselines3**. It is a library that provides ready-to-use implementations of reinforcement learning algorithms such as DQN and PPO, so you don't have to build them from scratch.

Without SB3, implementing DQN from scratch requires hundreds of lines of code. With SB3:

```python
model = DQN("CnnPolicy", env)
model.learn(total_timesteps=1_000_000)
model.save("my_model")
```

Three lines is all it takes.

**Who made it:** German Aerospace Center (DLR) + open source community  
**Current version:** Stable-Baselines3 (SB3) — PyTorch based  
**Previous versions:** SB1, SB2 — TensorFlow based (no longer used)

SB3 is not limited to games. It can be applied to any sequential decision-making problem:

| Field | Example |
|---|---|
| Games | Pong, Chess, Go |
| Robotics | Robot arm picking up objects |
| Self-driving | When to brake or steer |
| Finance | When to buy or sell stocks |
| Data centers | Server cooling optimization (used by Google) |

We use games because they are the easiest environment to test with — no physical damage on failure, fast to simulate, and results are easy to verify visually.

---

### gymnasium

gymnasium is a library that acts as a **standard bridge between AI models and game environments**.

**Who made it:**
- Originally created by OpenAI (the company behind ChatGPT) under the name `gym` in 2016
- OpenAI stopped maintaining it in 2022
- Farama Foundation took it over and renamed it `gymnasium`

```
OpenAI creates gym (2016)
        ↓
OpenAI stops maintaining it (2022)
        ↓
Farama Foundation takes over → gymnasium
```

**Why it matters:**  
Without gymnasium, every game (Pong, chess, robot simulators) would have a completely different way of communicating with the AI. gymnasium provides a standard interface — `reset()`, `step()`, `render()` — so any AI can connect to any environment the same way.

**gym vs gymnasium:**

| | gym | gymnasium |
|---|---|---|
| Creator | OpenAI | Farama Foundation |
| Status | Unmaintained since 2022 | Actively developed |
| NumPy 2.0 support | No | Yes |
| In this project | Installed for compatibility only | Actually used |

---

## How the Emulator and AI Exchange Data

Every frame of the game, this loop runs (~60 times per second):

```
EMULATOR                            AI MODEL
   │                                    │
   │  obs (84x84 pixels × 4 frames)     │
   │ ──────────────────────────────────▶│
   │                                    │ CNN processes pixels
   │                                    │ Computes Q-values
   │  action (a single number 0~5)      │
   │ ◀──────────────────────────────────│
   │ Move paddle                        │
   │ Update ball position               │
   │ Calculate score                    │
   │  reward (+1 / -1 / 0)              │
   │  obs (new screen)                  │
   │ ──────────────────────────────────▶│
   │                                    │ (repeat)
```

| Variable | Direction | Content |
|---|---|---|
| `obs` | Emulator → AI | 84x84 pixel screen, 4 frames stacked |
| `action` | AI → Emulator | A single number: 0=NOOP, 2=UP, 3=DOWN |
| `reward` | Emulator → AI | +1 (scored), -1 (missed), 0 (ongoing) |
| `done` | Emulator → AI | True = game over, False = keep going |

---

## Line-by-Line Code Explanation

```python
import ale_py
```
Loads the Atari game engine. The actual Pong game is inside this package.

```python
import gymnasium
gymnasium.register_envs(ale_py)
```
Registers all Atari games into gymnasium so that names like `PongNoFrameskip-v4` are recognized.

```python
from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
```
- `load_from_hub` — downloads a model from HuggingFace
- `DQN` — the DQN algorithm
- `make_atari_env` — creates an Atari game environment (made by SB3 team)
- `VecFrameStack` — stacks 4 consecutive frames into one input

```python
import warnings
warnings.filterwarnings("ignore")
```
Suppresses unnecessary warning messages at runtime.

---

```python
checkpoint = load_from_hub(
    repo_id="sb3/dqn-PongNoFrameskip-v4",
    filename="dqn-PongNoFrameskip-v4.zip",
)
```
Downloads the pretrained DQN model from HuggingFace.  
`repo_id` is the HuggingFace repository address, `filename` is the file to download.  
This model has already been trained for millions of steps — no training happens here.

---

```python
env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
```
Creates the Pong emulator. Each argument explained:

- `"PongNoFrameskip-v4"` — which game to create
- `n_envs=1` — create 1 game instance (can use 4 during training to speed things up)
- `seed=0` — fixes the random starting conditions so results are reproducible
- `render_mode="human"` — display the game on the monitor

```python
env = VecFrameStack(env, n_stack=4)
```
Wraps the emulator to stack 4 consecutive frames into one input.  
A single frame cannot tell the AI which direction the ball is moving.  
With 4 frames stacked, the AI can infer both direction and speed.

```
Single frame : ball is here — but where is it going?
4 frames     : ● → ● → ● → ● — moving right!
```

The AI input size becomes: **84 × 84 × 4** (width × height × frames)

---

```python
custom_objects = {
    "learning_rate": 1e-4,
    "lr_schedule": lambda _: 1e-4,
    "exploration_schedule": lambda _: 0.01,
    "optimize_memory_usage": False,
}
model = DQN.load(checkpoint, env=env, custom_objects=custom_objects)
```
The HuggingFace model was saved with an older version of SB3. Our installed version is newer, so some values cannot be read from the file. `custom_objects` overrides those values manually — "don't read these from the file, use what I give you instead."

| Key | Purpose | Why needed |
|---|---|---|
| `learning_rate` | How fast Q-values are updated during training | Cannot be read from old file |
| `lr_schedule` | How learning rate changes over time (fixed here) | Cannot be read from old file |
| `exploration_schedule` | Probability of random action (1% here) | Cannot be read from old file |
| `optimize_memory_usage: False` | Disables memory-saving mode | **Most important** — setting this to True causes a crash in newer SB3 |

Since we are only running the model (not training), `learning_rate`, `lr_schedule`, and `exploration_schedule` have no actual effect. They are just placeholder values to prevent load errors.

---

```python
obs = env.reset()
```
Initializes the game and receives the first screen from the emulator.

```python
while True:
```
Runs the game loop forever until Ctrl+C is pressed.

```python
    action, _ = model.predict(obs, deterministic=True)
```
The AI looks at the current screen (`obs`) and decides an action.  
`deterministic=True` means always pick the action with the highest Q-value — no random exploration.

```python
    obs, reward, done, info = env.step(action)
```
Sends the action to the emulator. The emulator moves the paddle, updates the ball, and returns the new screen, reward, and whether the game ended.

```python
    if done[0]:
        episode += 1
        score = info[0].get("episode", {}).get("r", "?")
        print(f"Episode {episode} 완료 - 점수: {score}", flush=True)
        obs = env.reset()
```
When the game ends (`done=True`), prints the score and starts a new game.
