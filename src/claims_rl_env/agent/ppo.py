import numpy as np


class PPO:
    def __init__(self, policy, clip=0.2, lr=0.001):
        self.policy = policy
        self.clip = clip
        self.lr = lr

    def update(self, trajectories):

        # unpack
        states, actions, old_probs, rewards = zip(*trajectories)

        rewards = np.array(rewards)

        # normalize rewards → acts like advantage
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8
        advantages = (rewards - mean) / std

        for state, action_idx, old_prob, adv in zip(states, actions, old_probs, advantages):

            probs = self.policy.get_probs()
            new_prob = probs[action_idx]

            ratio = new_prob / (old_prob + 1e-8)

            # clipped objective
            unclipped = ratio * adv
            clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip) * adv

            # PPO objective = maximize min(...)
            objective = min(unclipped, clipped)

            # gradient of log policy
            grad = self.policy.grad_log_prob(action_idx)
            grad = np.array(grad)

            # gradient ascent
            update = self.lr * objective * grad

            # safe update
            self.policy.params = self.policy.params + update

            # safety check 
            if self.policy.params.ndim != 1:
                raise ValueError(f"Params broke shape: {self.policy.params}")