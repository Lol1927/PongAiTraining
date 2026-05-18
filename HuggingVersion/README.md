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

Think of Q-value like a student deciding what to do before an exam:

| Situation | Action | Q-value (expected outcome) |
|---|---|---|
| Exam is tomorrow | Study hard | 9.5 — very likely to pass |
| Exam is tomorrow | Watch TV | 1.2 — very likely to fail |
| Exam is tomorrow | Sleep early | 6.0 — might help |

The student picks "Study hard" because it has the highest Q-value.
The DQN AI does the exact same thing — but for paddle movements in Pong.

A concrete Pong example:

| Situation | Action | Q-value |
|---|---|---|
| Ball is coming toward paddle | Move up | 8.2 — likely to hit the ball |
| Ball is coming toward paddle | Move down | 1.5 — likely to miss |
| Ball is moving away | Stay still | 7.0 — no need to move yet |

The AI always picks the action with the highest Q-value.

**How it learns:**

At the very start, the AI knows nothing — its Q-values are random guesses.
Here is exactly how it improves step by step:

```
Step 1: The ball is heading toward the paddle.
        AI guesses: "Move down" looks good (Q = 6.0)
        AI moves the paddle down.

Step 2: The paddle misses the ball. Opponent scores.
        AI receives a reward of -1 (punishment for missing).

Step 3: AI updates its memory:
        "In that situation, moving down gave me -1.
         I should lower the Q-value for moving down."
        → Q-value for "move down" in that situation: 6.0 → 2.1

Step 4: Next time the same situation appears,
        AI is less likely to move down.
```

This process repeats millions of times:
- Good action → reward +1 → Q-value goes up → AI does it more often
- Bad action  → reward -1 → Q-value goes down → AI avoids it

After millions of repetitions, the Q-values become accurate enough that the AI almost never misses the ball.

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
