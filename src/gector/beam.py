from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class Hypothesis:
    """One beam candidate for a single sentence."""
    tokens: List[str]               # current token sequence (with $START)
    score: float = 0.0             # cumulative log-prob
    tag_history: List[List[str]] = field(default_factory=list)
    is_finished: bool = False       # True once no edits remain

    def clone(self) -> "Hypothesis":
        return Hypothesis(
            tokens=self.tokens.copy(),
            score=self.score,
            tag_history=[t.copy() for t in self.tag_history],
            is_finished=self.is_finished
        )

@dataclass
class BeamState:
    """All active beams for one sentence."""
    beams: List[Hypothesis]
    sentence_id: int                # index into original srcs list

    def best(self) -> Hypothesis:
        return max(self.beams, key=lambda h: h.score)