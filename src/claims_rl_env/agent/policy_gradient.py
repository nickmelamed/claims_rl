import numpy as np


class PolicyGradient:
    def __init__(self, policy, lr=0.01):
        self.policy = policy
        self.lr = lr

    def update(self, trajectories):
        for state, action_idx, reward in trajectories:
            grad = self.policy.grad_log_prob(action_idx)
            self.policy.params += self.lr * grad * reward