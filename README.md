# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium
            break

    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        q_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(q_values)
    return V, policy, iterations

# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

V, policy, iteration = value_iteration(env)

# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: AVINASH T")
print("Register Number: 212223230026")
print("Value Iteration Completed")
print("Number of Iterations:", iteration)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()


```

---

## Output

<img width="581" height="292" alt="image" src="https://github.com/user-attachments/assets/169ac401-5d8a-47c3-bb2b-2c9b334ed9a6" />



## Result

The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were computed after convergence using the Bellman Optimality Equation.


## Inference

From this experiment, it is observed that the Value Iteration algorithm efficiently computes the optimal value of every state by repeatedly applying the Bellman Optimality Equation. Once the value function converges, the optimal policy is extracted by selecting the action with the highest expected return. This demonstrates how Dynamic Programming can solve finite Markov Decision Processes and determine the best sequence of actions for an agent.

