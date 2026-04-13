import random
from env.state import State
from env.actions import Actions


class ClaimEnv:
    def __init__(self, dataset, judge, curriculum):
        self.dataset = dataset
        self.judge = judge
        self.curriculum = curriculum
        self.state = None

    def reset(self):
        sample = self.curriculum.sample(self.dataset)
        self.state = State(
            claim=sample["claim"],
            evidence_pool=sample["evidence"]
        )
        return self.state

    def step(self, action, payload=None):
        s = self.state
        s.steps_taken += 1

        if action == Actions.SELECT:
            doc = next(e for e in s.evidence_pool if e.id == payload)
            if doc not in s.selected_evidence:
                s.selected_evidence.append(doc)

        elif action == Actions.REMOVE:
            s.selected_evidence = [
                e for e in s.selected_evidence if e.id != payload
            ]

        elif action == Actions.SUPPORT:
            s.debate_history.append("SUPPORT: " + payload)

        elif action == Actions.CONTRADICT:
            s.debate_history.append("CONTRADICT: " + payload)

        elif action == Actions.FINALIZE:
            reward = self.judge.evaluate(s)
            return s, reward, True, {}

        done = s.is_done()
        return s, 0.0, done, {}