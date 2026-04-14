import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.values = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.values)

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]

        self.values[action] += (reward - self.values[action]) / n