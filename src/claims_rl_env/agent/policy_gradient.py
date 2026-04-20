import numpy as np


class PolicyGradient:
    def __init__(self, policy, lr=0.01):
        self.policy = policy
        self.lr = lr

    def update(self, trajectories):
        rewards = [r for (_, _, r) in trajectories]
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8

        for _, action_idx, reward in trajectories:

            normalized_reward = (reward - mean) / std

            grad = self.policy.grad_log_prob(action_idx)
            grad = np.array(grad)

            update = self.lr * grad * normalized_reward

            self.policy.params = self.policy.params + update