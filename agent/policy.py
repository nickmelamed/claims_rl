import random
from env.actions import Actions


class RandomPolicy:
    def act(self, state):
        actions = [
            Actions.SELECT,
            Actions.SUPPORT,
            Actions.CONTRADICT,
            Actions.FINALIZE
        ]
        action = random.choice(actions)

        if action == Actions.SELECT:
            doc = random.choice(state.evidence_pool)
            return action, doc.id

        elif action in [Actions.SUPPORT, Actions.CONTRADICT]:
            return action, "Argument text"

        return action, None