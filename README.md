# PongAiTraining

A two-phase reinforcement learning project that trains and evaluates AI agents to play Atari Pong using Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms.

## Project Structure

```
PongAiTraining/
├── HuggingVersion/        # Phase 1: Pre-trained model inference
│   ├── phase1_test.py     # Run a pre-trained DQN agent from HuggingFace
│   ├── README.md
│   └── CODE_EXPLAINED.md
├── IndividualTraining/    # Phase 2: Train from scratch
│   ├── train.py           # DQN training script
│   ├── train_ppo.py       # PPO training script
│   ├── README.md
│   └── DQN_PROBLEM_AND_PPO.md
└── .gitignore
```

## Phases

### Phase 1 — Pre-trained Inference (HuggingVersion)
Downloads and runs `sb3/dqn-PongNoFrameskip-v4`, a DQN model pre-trained on millions of Atari frames from HuggingFace Hub. Used to understand agent behavior before custom training.

- **Environment:** `PongNoFrameskip-v4` (Gymnasium + ALE)
- **Model:** Stable-Baselines3 DQN via `huggingface-sb3`
- **Input:** 4 stacked grayscale frames for temporal context

### Phase 2 — Custom Training (IndividualTraining)
Trains a DQN agent from scratch, then switches to PPO after observing DQN instability on limited hardware.

- **Hardware target:** NVIDIA Jetson Orin Nano
- **Algorithms:** DQN → PPO (Stable-Baselines3)
- **Monitoring:** TensorBoard for reward curves and training metrics
- See `DQN_PROBLEM_AND_PPO.md` for analysis of why PPO outperformed DQN in this setup

## Key Concepts

| Concept | Description |
|---------|-------------|
| DQN | Assigns Q-values to actions; selects highest-value action each frame |
| PPO | Policy gradient method; more stable training than DQN on limited compute |
| Frame stacking | 4 consecutive frames stacked so the agent perceives ball velocity |
| Reward signal | +1 for scoring, -1 for conceding a point |

## Setup

```bash
pip install stable-baselines3 huggingface-sb3 gymnasium[atari] ale-py AutoROM shimmy
AutoROM --accept-license
```

**Phase 1 — run pre-trained agent:**
```bash
cd HuggingVersion
python phase1_test.py
```

**Phase 2 — train from scratch:**
```bash
cd IndividualTraining
python train_ppo.py   # recommended
# or
python train.py       # DQN baseline
```

## Tech Stack

- Python 3.x
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Gymnasium + ALE (Arcade Learning Environment)
- HuggingFace Hub (`huggingface-sb3`)
- TensorBoard
