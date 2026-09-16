# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

## NAME : AVINASH T
## REG NO : 212223230026
---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

Develop a Python program that applies the Value Iteration algorithm to the FrozenLake-v1 environment provided by Gymnasium. The algorithm should iteratively update the value of each state until convergence and then derive the optimal policy that maximizes the expected cumulative reward.


## Software Requirements

Python 3.x
Gymnasium
NumPy
Jupyter Notebook / Google Colab / VS Code


## Environment Description

The FrozenLake-v1 environment is a grid-world problem in which an agent must move from the Start (S) state to the Goal (G) while avoiding Holes (H).

Grid Used:

F F S F
F H H F
F F G H
F F F H
Where:

S – Start State
F – Frozen Surface (Safe)
H – Hole (Terminal State)
G – Goal State (Reward = 1)
The environment is stochastic (is_slippery=True), meaning the intended action may not always be executed.


## MDP Representation

An MDP is represented as:

MDP = (S, A, P, R, γ)

Where:

S = Set of states (16 states)
A = {Left, Down, Right, Up}
P(s'|s,a) = Transition probability
R(s,a,s') = Reward function
γ = 0.99 = Discount factor



## Theory

Value Iteration is a Dynamic Programming algorithm used to compute the optimal value function of an MDP.

It repeatedly updates the value of each state using the Bellman Optimality Equation:

[ V(s)=\max_a\sum_{s'}P(s'|s,a)\left[R(s,a,s')+\gamma V(s')\right] ]
