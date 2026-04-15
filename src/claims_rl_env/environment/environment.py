import random
import numpy as np

from claims_rl_env.environment.state import State, Evidence
from claims_rl_env.environment.actions import Actions
from claims_rl_env.judge.reward import RewardFunction


class ClaimEnv:
    def __init__(self, dataset):
        self.dataset = dataset
        self.state = None
        self.current_sample = None
        self.reward_fn = RewardFunction()

    def reset(self):
        self.current_sample = random.choice(self.dataset)

        self.state = State(
            claim=self.current_sample["claim"],
            evidence_pool=[Evidence(**e) for e in self.current_sample["evidence"]]
        )

        return self.state

    def step(self, action, payload):
        s = self.state
        s.steps_taken += 1

        # action handling
        if action == Actions.SELECT:
            doc = next((e for e in s.evidence_pool if e.id == payload), None)
            if doc and doc not in s.selected_evidence:
                s.selected_evidence.append(doc)

        elif action == Actions.REMOVE:
            s.selected_evidence = [
                e for e in s.selected_evidence if e.id != payload
            ]

        elif action == Actions.SUPPORT:
            if payload:
                s.debate_history.append("SUPPORT: " + str(payload))

        elif action == Actions.CONTRADICT:
            if payload:
                s.debate_history.append("CONTRADICT: " + str(payload))

        elif action == Actions.FINALIZE:
            # build the final output 
            reasoning = " ".join(s.debate_history)

            # penalty for 0 evidence 
            if len(s.selected_evidence) == 0:
                return s, -1.0, True, {}
            
            # penalty for not taking enough steps
            if s.steps_taken <= 2:
                return s, -0.5, False, {}

            # confidence heuristic
            confidence = min(1.0, len(s.selected_evidence) / 3)

            # use dataset ground truth if available
            if "label" in self.current_sample:
                true_score = float(self.current_sample["label"])
            else:
                # fallback heuristic
                true_score = (
                    np.mean([
                        1 if e.label == "support" else 0
                        for e in s.selected_evidence
                    ])
                    if s.selected_evidence else 0
                )

            final_output = {
                "reasoning": reasoning,
                "confidence": confidence,
                "true_score": true_score
            }

            reward = self.reward_fn.compute(s, final_output)

            # penalty for empty debate 
            if not s.debate_history:
                reward -= 0.3

            # reward for evidence use
            reward *= 0.2 * len(s.selected_evidence)

            return s, reward, True, {}

        # step limit termination
        if s.is_done():
            # penalize not finalizing
            return s, -0.2, True, {}

        # default step
        return s, 0.0, False, {}
