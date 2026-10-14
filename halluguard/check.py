"""Check an answer against its context, claim by claim."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .scorers import Scorer
from .split import Span, sentences, windows

SUPPORTED, UNSUPPORTED, CONTRADICTED = "supported", "unsupported", "contradicted"


@dataclass
class ClaimResult:
    text: str
    start: int
    end: int
    support: float  # best support over all context windows
    contradiction: float  # contradiction score of that best window
    verdict: str
    evidence: str  # the context window that supports it best


@dataclass
class Report:
    claims: list[ClaimResult] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        return any(c.verdict != SUPPORTED for c in self.claims)

    @property
    def score(self) -> float:
        """Lowest claim support: one unsupported claim makes the answer suspect."""
        return min((c.support for c in self.claims), default=1.0)

    def to_dict(self) -> dict:
        return {"hallucinated": self.hallucinated, "score": self.score, "claims": [asdict(c) for c in self.claims]}


def verdict(support: float, contradiction: float, threshold: float, contra_threshold: float = 0.5) -> str:
    if support >= threshold:
        return SUPPORTED
    if contradiction >= contra_threshold and contradiction > support:
        return CONTRADICTED
    return UNSUPPORTED


def score_claims(answer: str, context: str, scorer: Scorer, max_words: int = 120, min_claim_words: int = 4) -> list[tuple[Span, float, float, str]]:
    """Score every claim in `answer`: (span, best support, its contradiction, evidence)."""
    wins = windows(context, max_words=max_words)
    out = []
    for claim in sentences(answer, min_words=min_claim_words):
        scores = scorer.score(wins, claim.text)
        if not scores:
            out.append((claim, 0.0, 0.0, ""))
            continue
        best = max(range(len(scores)), key=lambda i: scores[i].support)
        contra = max(s.contradiction for s in scores)
        out.append((claim, scores[best].support, contra, wins[best]))
    return out


def check(answer: str, context: str, scorer: Scorer, threshold: float = 0.5, **kw) -> Report:
    rep = Report()
    for span, sup, contra, ev in score_claims(answer, context, scorer, **kw):
        rep.claims.append(ClaimResult(span.text, span.start, span.end, sup, contra, verdict(sup, contra, threshold), ev))
    return rep
