import numpy as np


class PPO:
    def __init__(self, policy, clip=0.2, lr=0.001):
        self.policy = policy
        self.clip = clip
        self.lr = lr

    def update(self, trajectories):

        rewards = [r for (_, _, r) in trajectories]
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8

        for state, action, old_prob, reward in trajectories:

            normalized_reward = (reward - mean) / std

            new_prob = self.policy.prob(state, action)

            ratio = new_prob / (old_prob + 1e-8)

            clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)

            loss = -min(ratio * normalized_reward, clipped * normalized_reward)

            grad = self.policy.grad(state, action)
            self.policy.params -= self.lr * loss * grad