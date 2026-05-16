# 🧗 CliffWalking — SARSA Agent

A Reinforcement Learning agent trained to navigate the **CliffWalking-v1** environment using tabular **SARSA** with ε-greedy exploration — built with NumPy and Gymnasium.

![CliffWalking Environment](./cliffwalking.png)

---

## How It Works

- A **Q-table** of shape `(48, 4)` stores state-action values
- Agent picks actions via **ε-greedy policy** (explore vs exploit)
- Q-values updated using the **SARSA (on-policy) update rule**:  
  `Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]`
- Unlike Q-Learning, SARSA updates using the **actual next action** taken, not the greedy max

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (α) | 0.5 |
| Discount factor (γ) | 0.99 |
| Epsilon (ε) | 0.1 |
| Episodes | 500 |
| Q-table size | 48 × 4 |

---

## Q-Learning vs SARSA

| | Q-Learning | SARSA |
|---|---|---|
| Type | Off-policy | On-policy |
| Update uses | `max Q(s',a')` | `Q(s', actual a')` |
| Behaviour | More aggressive | More cautious |

---

## Setup & Run

```bash
pip install gymnasium "gymnasium[toy-text]" numpy
```

Open and run [`SARSA.ipynb`](https://github.com/DeepanshuTevathiya/ML/blob/main/Reinforcement/cliffwalking_SARSA/SARSA.ipynb)

---

## Author

**Deepanshu Tevathiya** — AI / ML Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/deepanshu-tevathiya/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/DeepanshuTevathiya)
