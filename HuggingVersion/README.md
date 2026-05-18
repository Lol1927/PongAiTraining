# Phase 1: HuggingFace Pretrained Model Test

Downloads the `sb3/dqn-PongNoFrameskip-v4` model from HuggingFace and runs it on the Pong emulator.

## Setup

```bash
pip install stable-baselines3[extra] huggingface-sb3 gymnasium[atari] ale-py autorom shimmy gym
AutoROM --accept-license
```

## Run

```bash
python3 phase1_test.py
```

## How it works

- **Right paddle**: HuggingFace DQN AI (pretrained)
- **Left paddle**: Atari built-in opponent
- The model was trained with DQN on PongNoFrameskip-v4 and plays at near-perfect level
