import numpy as np


class PPO:
    def __init__(self, policy, clip=0.2, lr=0.001):
        self.policy = policy
        self.clip = clip
        self.lr = lr

    def update(self, trajectories):
        for state, action, old_prob, reward in trajectories:
            new_prob = self.policy.prob(state, action)

            ratio = new_prob / (old_prob + 1e-8)

            clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)

            loss = -min(ratio * reward, clipped * reward)

            grad = self.policy.grad(state, action)
            self.policy.params -= self.lr * loss * grad