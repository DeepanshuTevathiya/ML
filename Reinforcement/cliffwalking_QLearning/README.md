# 🧗 CliffWalking — Q-Learning Agent

A Reinforcement Learning agent trained to navigate the **CliffWalking-v1** environment using tabular **Q-Learning** with ε-greedy exploration — built with NumPy and Gymnasium.

![CliffWalking Environment](./cliffwalking.png)

---

## How It Works

- A **Q-table** of shape `(48, 4)` stores state-action values
- Agent picks actions via **ε-greedy policy** (explore vs exploit)
- Q-values updated each step using the **Bellman equation**:  
  `Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',a') − Q(s,a)]`
- Every 50 episodes, the environment renders visually to track progress

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

## Setup & Run

```bash
pip install gymnasium numpy
```

Open and run [`Q_Learning.ipynb`](https://github.com/DeepanshuTevathiya/ML/blob/main/Reinforcement/cliffwalking_QLearning/Q_Learning.ipynb)

---

## Author

**Deepanshu Tevathiya** — AI / ML Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/deepanshu-tevathiya/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/DeepanshuTevathiya)
