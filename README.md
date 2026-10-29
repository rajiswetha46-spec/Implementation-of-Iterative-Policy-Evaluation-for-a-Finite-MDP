# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP

### Name: Swetha R
### Reg No: 212225100055
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python


# -------------------------------------------------
# Policy Evaluation Function
# -------------------------------------------------
def policy_evaluation(env, policy, gamma=0.9, theta=1e-8):
    """
    Performs iterative policy evaluation using the Bellman expectation equation.

    Parameters:
        env    : Gymnasium FrozenLake environment
        policy : Fixed policy to be evaluated
        gamma  : Discount factor
        theta  : Convergence threshold

    Returns:
        V         : Estimated state-value function
        iteration : Number of iterations used for convergence
    """
    V = np.zeros(env.observation_space.n)
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            v = V[s]
            new_value = 0

            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, terminated in env.P[s][a]:
                    new_value += action_prob * prob * (
                        reward + gamma * V[next_state] * (not terminated)
                    )

            V[s] = new_value
            delta = max(delta, abs(v - V[s]))

        iteration += 1

        if delta < theta:
            break






    return V, iteration

# -------------------------------------------------
# Display Output
# -------------------------------------------------
# Run policy evaluation
V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name: swetha R")
print("Register Number: 212225100055")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("Name: Swetha R")
print("Register Number: 212225100055")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))
print("Gamma : ",gamma)
env.close()


```

---

## Output

<img width="582" height="385" alt="Screenshot 2026-10-23 105121" src="https://github.com/user-attachments/assets/1755c0b0-463b-42b2-a403-503c39cf54e9" />
<img width="597" height="381" alt="Screenshot 2026-10-23 105229" src="https://github.com/user-attachments/assets/169ebaac-8d2b-4d05-9b96-29092714bd63" />

---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference:

The random policy selects all four actions (Left, Down, Right, and Up) with equal probability (0.25) in every state.

For γ = 0.9, future rewards are discounted more heavily, resulting in lower state values. The highest state value obtained is 0.3915, and the algorithm converges in 38 iterations.

For γ = 0.99, future rewards are given greater importance, resulting in higher state values across the grid. The highest state value increases to 0.4336, and the algorithm requires 54 iterations to converge.

Thus, increasing the discount factor from 0.9 to 0.99 increases the state values and the number of iterations required for convergence, as the agent places more emphasis on long-term rewards. Terminal states (goal and holes) do not accumulate future rewards after being reached.




---


