# Opengym+RLGlue_ReinforcementLearning


## Demonstration

https://github.com/user-attachments/assets/788b38a7-cef0-4a6e-8d4d-079dcf8c83ed



<div align="left">
  <h3>PPO-Episode 500</h3>
  <video src="[https://github.com/YOUR-VIDEO-LINK-HERE](https://github.com/user-attachments/assets/788b38a7-cef0-4a6e-8d4d-079dcf8c83ed
).mp4" width="400" controls></video>
</div>

## Performance
![PPO_performance](results/ppo_performance.png)
![Q_learning_performance](results/training_performance.png)


## Implementaion and theory
A significant portion of the theory and logs below are derived from **Fundamentals of Reinforcement Learning** offered by the University of Alberta & Alberta Machine Intelligence Institute (Amii), taught by Martha White and Adam White.

* Formalize problems as Markov Decision Processes (MDPs).
* Understand basic exploration methods and the exploration/exploitation tradeoff.
* Use value functions as a general-purpose tool for optimal decision-making.
* Implement dynamic programming as an efficient solution approach.

### 1. The Epsilon-Greedy Agent

The Epsilon-Greedy agent balances exploration and exploitation. Here is the core logic:

```python
# For an epsilon-greedy agent: 
if np.random.random() < self.epsilon:
    # Exploration with probability: epsilon
    # Find action arm completely at random
    current_action = np.random.randint(0, len(self.q_values))
else:
    # Exploitation: pick the best known action
    current_action = np.argmax(self.q_values)
```

**Main Simulation Parameters:**
* **Number of Independent runs:** 200
* **Time steps per run:** 1000
* **Exploration Probability ($\epsilon$):** 10% (0.1)
![.1-epsilon-exploration_greedy_agent](0.1-epsilon-exploration_greedy_agent-1.png)

---

### 2. Managing Randomness & Stochasticity

A random simulation with an epsilon-greedy agent relies on several random elements:
* The decision to explore.
* The random action chosen initially.
* Tie-breaking (when multiple actions have the same `argmax` value).
* Reward distribution (randomly sampled from a Gaussian).

**Wiping out random fluctuation:**
To wash out noise and see the "big picture," we rely on the power of averaging.
* We average across 200 independent runs: `np.mean(all_reward, axis=0)`
* Tracking the cumulative average helps smooth out the stochasticity of individual seeds.

**Are statistical significance tests needed?** No. Because we have access to simulators for our experiments, we use the simpler strategy of running for a large number of runs and ensuring that the confidence intervals do not overlap.

**Performance Visualization (Averaged vs. Individual Runs)**

* **Averaged performance:** Shows the smooth trend of the 0.1 epsilon agent over many runs.
* **The inherent noise:** Becomes obvious when viewing only two individual runs.
![2-Individuakruns_0.1-epsilon-exploration](2-Individuakruns_0.1-epsilon-exploration_greedy_agent-1.png)


---

### 3. Comparing Values of Epsilon

Testing epsilons = [0.0, 0.01, 0.1, 0.4] revealed that 10% exploration (0.1) is the best performing parameter for this setup.

* **0.1 (10%):** Optimal balance.
* **0.01 (1%):** Explores too little and takes too long to find the best arm; still in the discovery phase by step 1000.
* **0.4 (40%):** Spends too much time choosing randomly, picking suboptimal arms too frequently.
![change_epsilon](change_epsilon.png)

---

### 4. The Effect of Step Size

We evaluated how different step sizes impact the agent's ability to lock onto the true expected value.

**Tested step sizes:** [0.01, 0.1, 0.5, 1.0, $1/N(A)$]

```python
# Measuring the amount of time the best action is taken
if action == best_arm:
    best_action_chosen.append(1)
else:
    best_action_chosen.append(0)

if run == 0:
    q_values[step_size].append(np.copy(rl_glue.agent.q_values))
    
best_actions[step_size].append(best_action_chosen)
ax.plot(np.mean(best_actions[step_size], axis=0))
```

**Step Size Performance Comparison**

Conclusions on Step Size for 1000 steps: $1/N(A) > 0.1 > 0.5 > 1.0 > 0.01$

* **$1/N(A)$ (Decaying step size):** Performs best in stationary environments. It moves quickly at first but reduces later, making it less susceptible to the stochasticity of rewards.
* **0.5 & 1.0 (Large step sizes):** Overcorrect and oscillate. Highly susceptible to stochasticity.
* **0.1 (Moderate step size):** Moves steadily and does not oscillate wildly.

![step_size_comparison-runs](step_size_comparison-runs_0.1-epsilon-exploration_greedy_agent-1.png)


> **Note:** While a decaying step size ($1/N(A)$) converges without oscillating, it cannot adapt to changes in the environment. Nonstationarity is a common feature of online reinforcement learning problems, where a constant step size (like 0.1) often becomes necessary to adapt to shifting reward distributions.
## Reference 
https://github.com/seungeunrho/minimalRL

# Discussions
Q1> Compare bandits to supervised learning

A1> "Supervised Learning and the Bandit Problem are similar in decision optimization, as both use input-output mapping and probability distributions (like Softmax) to guide their choices without a pre-existing physical model to follow. However, they differ in how they use those probabilities: Supervised Learning uses them to match a known answer, whereas the Bandit Problem uses them for an active search, balancing exploration and exploitation to discover the best action through trial and error."