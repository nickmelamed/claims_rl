# TODO: action masking? 

import numpy as np
from claims_rl_env.environment.actions import Actions, ACTIONS


class SoftmaxPolicy:
    def __init__(self, n_actions):
        self.actions = ACTIONS
        self.n_actions = n_actions
        self.params = np.zeros(n_actions)  # logits

        finalize_idx = self.actions.index(Actions.FINALIZE)
        self.params[finalize_idx] = -2.0

    def get_probs(self):
        exp = np.exp(self.params)
        return exp / np.sum(exp)

    def act(self, state):
        probs = self.get_probs()

        # mask FINALIZE early
        if state.steps_taken < 2:
            finalize_idx = self.actions.index(Actions.FINALIZE)
            probs[finalize_idx] = 0
            probs = probs / np.sum(probs)

        # epsilon-greedy exploration
        if np.random.rand() < 0.3:
            idx = np.random.choice(self.n_actions)
        else:
            idx = np.random.choice(self.n_actions, p=probs)

        action = self.actions[idx]

        # payload logic
        if action == Actions.SELECT:
            doc = np.random.choice(state.evidence_pool)
            return action, doc.id

        elif action in [Actions.SUPPORT, Actions.CONTRADICT]:
            return action, "Generated argument"

        return action, None

    def grad_log_prob(self, action_idx):
        probs = self.get_probs()
        grad = -probs
        grad[action_idx] += 1
        return grad