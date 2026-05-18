# Phase 1: HuggingFace Pretrained Model Test

## What is HuggingFace?

HuggingFace is a platform for sharing and downloading AI models — similar to GitHub, but for trained models instead of code.
Instead of training a model ourselves (which takes hours), we can download a model that someone else already trained and use it immediately.

In this project, we download the model `sb3/dqn-PongNoFrameskip-v4` uploaded by the Stable-Baselines3 team, which has already been trained for millions of steps on Pong.

---

## What is DQN?

DQN (Deep Q-Network) is a reinforcement learning algorithm that teaches an AI to play games by trial and error.

**Core idea — Q-value:**
The AI assigns a score (Q-value) to every possible action in a given situation, then picks the action with the highest score.

```
Current screen → AI → Q-value per action
                       ├─ Move up:    8.2  ← highest → chosen
                       ├─ Move down:  3.1
                       └─ Stay still: 1.5
```

**How it learns:**
1. Look at the screen and pick an action
2. Get a reward (+1 for scoring, -1 for missing)
3. Remember which actions led to high rewards
4. Repeat millions of times → Q-values become accurate

**Why "Deep":**
The Q-values are computed by a CNN (Convolutional Neural Network) that reads raw pixel data directly from the screen.

```
Game screen (pixels) → CNN → Q-values → Action
     84x84                    up / down / stay
```

---

## What is PongNoFrameskip-v4?

`PongNoFrameskip-v4` is the name of the Pong game environment provided by the Atari Learning Environment (ALE).

| Part | Meaning |
|---|---|
| `Pong` | The classic Atari Pong game |
| `NoFrameskip` | Every single frame is processed (no frames are skipped) |
| `v4` | Version 4 of this environment |

The AI receives 4 consecutive frames stacked together as input so it can understand the direction and speed of the ball — something that is impossible to determine from a single frame alone.

```
Frame 1 + Frame 2 + Frame 3 + Frame 4 → AI input (84x84x4)
```

---

## What is the Atari Built-in Opponent?

The left paddle in Pong is controlled by a simple rule-based AI that is built into the Atari emulator. It was not trained with machine learning — it just follows the ball using fixed rules. It is relatively easy to beat.

```
┌─────────────────────┐
│                     │
│ |              |    │
│ |      ● →     |    │
│ |              |    │
│                     │
└─────────────────────┘
  ↑                ↑
Left paddle      Right paddle
(Atari built-in  (Our DQN AI
 opponent)        from HuggingFace)
```

---

## Setup

```bash
pip install stable-baselines3[extra] huggingface-sb3 gymnasium[atari] ale-py autorom shimmy gym
AutoROM --accept-license
```

## Run

```bash
python3 phase1_test.py
```
