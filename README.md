# Reinforcement Learning — Example Source Code

Source code for the programs in the English edition of *Reinforcement Learning* (HANBIT Academy). Each file corresponds to one numbered **Program** in the book: Program 9-5 is `ch09/program_9-5.py`.

- Line numbers match the book, so the "Line N" explanations in the text apply directly to these files.
- Every file is a complete script. Programs that the book presents as "identical to Program X except for the code below" are provided here in full.
- Indentation uses 4 spaces throughout.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` does not pin versions. Environments used in the book and the Gymnasium extras they need:

| Environments | Extra |
|---|---|
| FrozenLake-v1, Blackjack-v1 | `toy-text` |
| CartPole-v1, MountainCarContinuous-v0 | `classic-control` |
| Hopper-v4, InvertedPendulum-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4, Humanoid-v4 | `mujoco` |
| ALE/Pong-v5 | `atari` |

## Running order

Some programs load files saved by an earlier program. Run the producing program first.

| Program | Needs file | Produced by |
|---|---|---|
| 5-5 | `f5-4.pkl` | Program 5-4 |
| 5-6 | `f5-4.pkl` | Program 5-4 |
| 5-7 | `f5-4.pkl` | Program 5-4 |
| 5-10 | `f5-9.pkl` | Program 5-9 |
| 6-3 | `f6-2.keras` | Program 6-2 |
| 6-5 | `f6-4.keras` | Program 6-4 |
| 6-6 | `f6-4.keras` | Program 6-4 |
| 7-6 | `f7-5.keras` | Program 7-5 |
| 8-5 | `f8-2.keras` | Program 8-2 or Program 8-3 or Program 8-4 |
| 8-7 | `f8-6.keras` | Program 8-6 or Program 8-8 |
| 9-6 | `f9-5.weights.h5` | Program 9-5 or Program 9-7 |
| 9-8 | `f9-7.weights.h5` | Program — |
| 10-2 | `f10-1.weights.h5` | Program 10-1 |
| 10-4 | `f10-3.weights.h5` | Program 10-3 |
| 10-6 | `f10-5.weights.h5` | Program 10-5 |
| 10-8 | `f10-7.weights.h5` | Program 10-7 |
| 10-10 | `f10-9.weights.h5` | Program 10-9 |
| 11-2 | `f11-1.keras` | Program 11-1 |
| 11-4 | `f11-3.keras` | Program 11-3 |
| 11-6 | `f11-5.keras` | Program 11-5 |
| 12-1 | `f10-1.weights.h5` | Program 10-1 |
| 12-2 | `f12-1.keras` | Program 12-1 |
| 12-3 | `f10-1.weights.h5` | Program 10-1 |
| 12-4 | `f12-3.keras` | Program 12-3 |

## Programs


### Chapter 2

| Program | File | Title |
|---|---|---|
| 2-1 | [`ch02/program_2-1.py`](ch02/program_2-1.py) | Running MDP cycle for FrozenLake task |
| 2-2 | [`ch02/program_2-2.py`](ch02/program_2-2.py) | Estimating the expected return of random policy in FrozenLake |
| 2-3 | [`ch02/program_2-3.py`](ch02/program_2-3.py) | Estimating the expected return of optimal policy in FrozenLake |

### Chapter 3

| Program | File | Title |
|---|---|---|
| 3-1 | [`ch03/program_3-1.py`](ch03/program_3-1.py) | Evaluating random policy using Bellman equation (FrozenLake task) |
| 3-2 | [`ch03/program_3-2.py`](ch03/program_3-2.py) | Policy iteration algorithm for FrozenLake task |
| 3-3 | [`ch03/program_3-3.py`](ch03/program_3-3.py) | Value iteration algorithm for FrozenLake task |
| 3-4 | [`ch03/program_3-4.py`](ch03/program_3-4.py) | Checking MDP transition probability distributions for FrozenLake task |
| 3-5 | [`ch03/program_3-5.py`](ch03/program_3-5.py) | Value iteration algorithm for FrozenLake task (stochastic environment) |
| 3-6 | [`ch03/program_3-6.py`](ch03/program_3-6.py) | Value iteration algorithm for FrozenLake task (stochastic environment with increased intended direction probability) |

### Chapter 4

| Program | File | Title |
|---|---|---|
| 4-1 | [`ch04/program_4-1.py`](ch04/program_4-1.py) | Episode generation in the FrozenLake task |
| 4-2 | [`ch04/program_4-2.py`](ch04/program_4-2.py) | Episode generation and rendering in the CartPole task |
| 4-3 | [`ch04/program_4-3.py`](ch04/program_4-3.py) | Analysis of expected returns from pure exploration vs ε-greedy in multi-armed |
| 4-4 | [`ch04/program_4-4.py`](ch04/program_4-4.py) | Performance analysis of exploitation-focused, ε-greedy, and UCB in large search |
| 4-5 | [`ch04/program_4-5.py`](ch04/program_4-5.py) | Policy evaluation with Monte Carlo methods (state-value function calculation) |
| 4-6 | [`ch04/program_4-6.py`](ch04/program_4-6.py) | Policy evaluation with Monte Carlo methods (action-value function calculation) |
| 4-7 | [`ch04/program_4-7.py`](ch04/program_4-7.py) | Policy learning with Monte Carlo methods (ε-soft applied) |

### Chapter 5

| Program | File | Title |
|---|---|---|
| 5-1 | [`ch05/program_5-1.py`](ch05/program_5-1.py) | Policy learning with Q-learning (ε-greedy applied) |
| 5-2 | [`ch05/program_5-2.py`](ch05/program_5-2.py) | Learning curves of Q-learning |
| 5-3 | [`ch05/program_5-3.py`](ch05/program_5-3.py) | Blackjack game simulation |
| 5-4 | [`ch05/program_5-4.py`](ch05/program_5-4.py) | Learning Blackjack task with Q-learning |
| 5-5 | [`ch05/program_5-5.py`](ch05/program_5-5.py) | Blackjack game play |
| 5-6 | [`ch05/program_5-6.py`](ch05/program_5-6.py) | Profit comparison between random policy and Q function learned through Q-learning |
| 5-7 | [`ch05/program_5-7.py`](ch05/program_5-7.py) | Q Function analysis |
| 5-8 | [`ch05/program_5-8.py`](ch05/program_5-8.py) | Finding state variable ranges for CartPole task |
| 5-9 | [`ch05/program_5-9.py`](ch05/program_5-9.py) | Learning CartPole task using Q-learning |
| 5-10 | [`ch05/program_5-10.py`](ch05/program_5-10.py) | CartPole testing using Q-learning results |

### Chapter 6

| Program | File | Title |
|---|---|---|
| 6-1 | [`ch06/program_6-1.py`](ch06/program_6-1.py) | Measuring neural network generalization capability with MNIST data |
| 6-2 | [`ch06/program_6-2.py`](ch06/program_6-2.py) | Neural network-based Q-learning (CartPole task) |
| 6-3 | [`ch06/program_6-3.py`](ch06/program_6-3.py) | CartPole task testing |
| 6-4 | [`ch06/program_6-4.py`](ch06/program_6-4.py) | Blackjack learning |
| 6-5 | [`ch06/program_6-5.py`](ch06/program_6-5.py) | Blackjack profit estimation |
| 6-6 | [`ch06/program_6-6.py`](ch06/program_6-6.py) | Q function interpretation |

### Chapter 7

| Program | File | Title |
|---|---|---|
| 7-1 | [`ch07/program_7-1.py`](ch07/program_7-1.py) | CartPole learning using replay memory |
| 7-2 | [`ch07/program_7-2.py`](ch07/program_7-2.py) | CartPole learning using replay memory and behavior neural network |
| 7-3 | [`ch07/program_7-3.py`](ch07/program_7-3.py) | Generalization ability of deep learning models |
| 7-4 | [`ch07/program_7-4.py`](ch07/program_7-4.py) | Pong Atari game animation |
| 7-5 | [`ch07/program_7-5.py`](ch07/program_7-5.py) | Pong Atari game learning using DQN |
| 7-6 | [`ch07/program_7-6.py`](ch07/program_7-6.py) | Playing Pong Atari game |

### Chapter 8

| Program | File | Title |
|---|---|---|
| 8-1 | [`ch08/program_8-1.py`](ch08/program_8-1.py) | Gradient calculation and model learning using TensorFlow |
| 8-2 | [`ch08/program_8-2.py`](ch08/program_8-2.py) | CartPole learning using REINFORCE ([Algorithm 8-1] for time step mode) |
| 8-3 | [`ch08/program_8-3.py`](ch08/program_8-3.py) | CartPole learning using REINFORCE ([Algorithm 8-2] for episode mode) |
| 8-4 | [`ch08/program_8-4.py`](ch08/program_8-4.py) | CartPole learning using REINFORCE (learning neural networks multiple times using a single episode) |
| 8-5 | [`ch08/program_8-5.py`](ch08/program_8-5.py) | CartPole testing using REINFORCE |
| 8-6 | [`ch08/program_8-6.py`](ch08/program_8-6.py) | Learning InvertedPendulum using REINFORCE |
| 8-7 | [`ch08/program_8-7.py`](ch08/program_8-7.py) | InvertedPendulum testing |
| 8-8 | [`ch08/program_8-8.py`](ch08/program_8-8.py) | Learning InvertedPendulum using REINFORCE (array operation version) |

### Chapter 9

| Program | File | Title |
|---|---|---|
| 9-1 | [`ch09/program_9-1.py`](ch09/program_9-1.py) | Advantage function implementations: Monte Carlo, n-step, and GAE |
| 9-2 | [`ch09/program_9-2.py`](ch09/program_9-2.py) | Advantage function implementations: Monte Carlo and GAE |
| 9-3 | [`ch09/program_9-3.py`](ch09/program_9-3.py) | Learning the CartPole task using A2C: shared network |
| 9-4 | [`ch09/program_9-4.py`](ch09/program_9-4.py) | Learning the CartPole task using A2C: separate networks |
| 9-5 | [`ch09/program_9-5.py`](ch09/program_9-5.py) | Learning the InvertedPendulum task using A2C |
| 9-6 | [`ch09/program_9-6.py`](ch09/program_9-6.py) | Testing the InvertedPendulum task |
| 9-7 | [`ch09/program_9-7.py`](ch09/program_9-7.py) | Learning the MountainCarContinuous task using A2C |
| 9-8 | [`ch09/program_9-8.py`](ch09/program_9-8.py) | Testing the MountainCarContinuous task |

### Chapter 10

| Program | File | Title |
|---|---|---|
| 10-1 | [`ch10/program_10-1.py`](ch10/program_10-1.py) | Learning Hopper task using PPO-clip |
| 10-2 | [`ch10/program_10-2.py`](ch10/program_10-2.py) | Hopper task testing |
| 10-3 | [`ch10/program_10-3.py`](ch10/program_10-3.py) | Learning Walker2d task using PPO-clip |
| 10-4 | [`ch10/program_10-4.py`](ch10/program_10-4.py) | Walker2d task testing |
| 10-5 | [`ch10/program_10-5.py`](ch10/program_10-5.py) | Learning HalfCheetah task using PPO-clip |
| 10-6 | [`ch10/program_10-6.py`](ch10/program_10-6.py) | HalfCheetah task testing |
| 10-7 | [`ch10/program_10-7.py`](ch10/program_10-7.py) | Learning Ant task using PPO-clip |
| 10-8 | [`ch10/program_10-8.py`](ch10/program_10-8.py) | Ant task testing |
| 10-9 | [`ch10/program_10-9.py`](ch10/program_10-9.py) | Learning Humanoid task using PPO-clip |
| 10-10 | [`ch10/program_10-10.py`](ch10/program_10-10.py) | Humanoid task |

### Chapter 11

| Program | File | Title |
|---|---|---|
| 11-1 | [`ch11/program_11-1.py`](ch11/program_11-1.py) | Learning the Hopper task using DDPG |
| 11-2 | [`ch11/program_11-2.py`](ch11/program_11-2.py) | Testing the Hopper task (DDPG) |
| 11-3 | [`ch11/program_11-3.py`](ch11/program_11-3.py) | Learning the Hopper task using TD3 |
| 11-4 | [`ch11/program_11-4.py`](ch11/program_11-4.py) | Testing the Hopper task (TD3) |
| 11-5 | [`ch11/program_11-5.py`](ch11/program_11-5.py) | Learning the Hopper task using SAC |
| 11-6 | [`ch11/program_11-6.py`](ch11/program_11-6.py) | Testing the Hopper task (SAC) |

### Chapter 12

| Program | File | Title |
|---|---|---|
| 12-1 | [`ch12/program_12-1.py`](ch12/program_12-1.py) | Learning the Hopper task with vanilla BC |
| 12-2 | [`ch12/program_12-2.py`](ch12/program_12-2.py) | Testing vanilla BC |
| 12-3 | [`ch12/program_12-3.py`](ch12/program_12-3.py) | Learning the Hopper task with DAgger |
| 12-4 | [`ch12/program_12-4.py`](ch12/program_12-4.py) | Testing DAgger |
