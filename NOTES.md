# Q-learning
epsilon-greedy policy
  epsilon = 1/k (k=timestep)
  Policy exploration
  Exploration decreases as policy converges to optimal policy

# PID augmented with PPO
By continually projecting back to PID, the PPO network will not cease to
`explore`. Exploration vs exploitation is a problem with RL algorithms. The PPO
network has a different relationship with local minima as well because it is
allowed to `climb out` by being projected to a linear control method.
