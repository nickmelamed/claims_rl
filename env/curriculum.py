import random


class Curriculum:
    def __init__(self):
        self.level = 0  # 0=easy, 1=medium, 2=hard

    def sample(self, dataset):
        filtered = [
            d for d in dataset
            if self._match_difficulty(d["difficulty"])
        ]
        return random.choice(filtered)

    def _match_difficulty(self, diff):
        mapping = {"easy": 0, "medium": 1, "hard": 2}
        return mapping[diff] <= self.level

    def update(self, performance):
        if performance > 0.8:
            self.level = min(2, self.level + 1)