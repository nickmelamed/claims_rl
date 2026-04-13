from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Evidence:
    id: int
    text: str
    label: str  # "support", "contradict", "neutral", "adversarial"


@dataclass
class State:
    claim: str
    evidence_pool: List[Evidence]
    selected_evidence: List[Evidence] = field(default_factory=list)
    debate_history: List[str] = field(default_factory=list)
    steps_taken: int = 0
    max_steps: int = 10

    def is_done(self):
        return self.steps_taken >= self.max_steps