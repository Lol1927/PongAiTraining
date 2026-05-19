# DQN Problem & Why We Switched to PPO

## What Happened with DQN

During training with `train.py` (DQN), the Jetson Orin Nano forcibly rebooted mid-training.
Not a single checkpoint was saved — meaning the crash happened before 50,000 steps.

### Root Cause: Replay Buffer Memory Overflow

DQN stores every experience (game screen) in a **Replay Buffer** and randomly samples from it during learning.

```
buffer_size = 100,000 experiences
Each experience = 84 x 84 pixels x 4 frames x 4 bytes
Total = ~5.2GB
```

The Jetson Orin Nano does not have enough RAM to hold 5.2GB for the buffer alone.
The system ran out of memory and rebooted itself.

---

## Two Possible Solutions

### Option 1: Reduce buffer_size (stay with DQN)

```python
# Change this:
buffer_size=100_000   # ~5.2GB

# To this:
buffer_size=10_000    # ~520MB
```

**Pros:** Simple one-line fix, no algorithm change needed.  
**Cons:** Smaller buffer = less diverse past experiences = lower learning quality.
DQN relies on having a large variety of past experiences to sample from.
Reducing the buffer means the AI forgets older experiences faster and may converge to a suboptimal policy.

### Option 2: Switch to PPO

PPO has **no replay buffer at all**.
Instead of storing experiences, it plays a short batch, learns from it immediately, then throws it away.

```
PPO memory usage: ~200MB
DQN memory usage: ~5.2GB
```

**Pros:** Drastically lower memory usage. No risk of OOM crash on Jetson.  
**Cons:** Requires switching the algorithm entirely (but not much harder to use).

---

## Why We Chose PPO

We chose to switch to PPO rather than reduce the buffer size for one key reason:

> **Reducing buffer_size hurts learning quality. PPO avoids the problem entirely.**

DQN's strength comes from its large, diverse replay buffer. Cut the buffer to 10,000 and you lose much of that diversity — the AI may learn slower or reach a worse final performance.

PPO does not use a replay buffer by design. It was built to work efficiently with small, fresh batches of experience. On memory-constrained hardware like the Jetson Orin Nano, PPO is the better fit.

---

## What is PPO?

PPO stands for **Proximal Policy Optimization**.

### The Core Idea

Instead of learning *the value of each action* (like DQN), PPO learns a **policy** directly — a probability distribution over actions.

```
Game screen  →  Policy  →  UP: 65%, STAY: 25%, DOWN: 10%
                            Roll dice → take action
```

The goal of training: increase the probability of actions that led to good outcomes, decrease the probability of actions that led to bad outcomes.

---

### Actor-Critic Architecture

PPO uses two networks working together:

```
                ┌──────────────────┐
  Game screen → │   CNN Backbone   │  (shared image recognition)
                └────────┬─────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
     ┌─────────────┐           ┌─────────────┐
     │    ACTOR    │           │   CRITIC    │
     └─────────────┘           └─────────────┘
            │                         │
   Outputs action probabilities   Outputs a score
   "Go UP with 65% chance"        "This situation is worth +2.3"
```

- **Actor**: Decides what to do (outputs probabilities)
- **Critic**: Evaluates how good the current situation is (outputs a single number)

The Critic does **not** change the probabilities. It only judges.
The Actor reads the Critic's judgement and adjusts its own probabilities.

---

### Advantage: "Was this better or worse than expected?"

The Critic's main job is to compute the **Advantage**:

```
Advantage = Actual reward received - Critic's expected reward

Example:
  Critic predicted: +1.5 points
  Actually got:     +3.0 points
  Advantage:        +1.5  →  "Better than expected! Increase probability of this action."

Example:
  Critic predicted: +2.0 points
  Actually got:     +0.5 points
  Advantage:        -1.5  →  "Worse than expected. Decrease probability of this action."
```

---

### Why "Proximal"? The Clipping Trick

A naive policy gradient would try to make large updates when an action goes well:

```
Action was great → "Change UP probability from 30% to 90%!"
Result: Policy breaks for other situations → performance collapses
```

PPO prevents this with a **clip**:

```
clip_range = 0.2

Allowed change: at most ±20% relative to old probability
30% → max 36%  (30% × 1.2)
```

This is the "Proximal" in PPO — keep the new policy close (proximal) to the old one.
Small, stable steps instead of risky leaps.

---

### One Full Training Cycle (Rollout)

```
Step 1: Play 128 steps across 8 environments = 1,024 experiences collected
Step 2: Critic scores each experience → compute Advantage for each
Step 3: Actor updates probabilities (with clip applied), repeated 4 times (n_epochs)
Step 4: Discard all 1,024 experiences
Step 5: Go back to Step 1
```

This cycle repeats until 1,000,000 total steps are reached.

---

### DQN vs PPO Summary

| | DQN | PPO |
|---|---|---|
| Learns | Q-values (action values) | Policy (action probabilities) |
| Memory | Replay buffer (~5.2GB) | Rollout buffer (~200MB) |
| Experience reuse | Sampled thousands of times | Used 4 times then discarded |
| Exploration | ε-greedy (random chance) | Entropy bonus |
| Stability | Can be unstable | Stable due to clipping |
| Jetson compatibility | OOM crash | Works fine |
