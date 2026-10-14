"""Scorers answer one question: how well does a piece of context support a claim?

Each scorer returns, for every (premise, claim) pair, a `Score` with
`support` (0..1) and `contradiction` (0..1).

- NLIScorer      a cross-encoder trained on natural language inference (the main method)
- CosineScorer   embedding similarity (the common baseline, and its weakness)
- OverlapScorer  share of the claim's content words found in the context (no model)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Score:
    support: float
    contradiction: float = 0.0


class Scorer(Protocol):
    name: str

    def score(self, premises: Sequence[str], claim: str) -> list[Score]: ...


def score_many(scorer, premises: Sequence[str], claims: Sequence[str]) -> list[list[Score]]:
    """Scores for every (premise, claim) pair, one list per claim.

    Uses the scorer's batched `score_many` when it has one (one model call per
    answer instead of one per sentence), otherwise falls back to `score`.
    """
    if hasattr(scorer, "score_many"):
        return scorer.score_many(premises, claims)
    return [scorer.score(premises, c) for c in claims]


class NLIScorer:
    """Cross-encoder NLI. Default: cross-encoder/nli-deberta-v3-base."""

    def __init__(self, model: str = "cross-encoder/nli-deberta-v3-base", batch_size: int = 32, device: str | None = None):
        from sentence_transformers import CrossEncoder  # imported lazily: heavy

        self.name = f"nli:{model.split('/')[-1]}"
        self.model = CrossEncoder(model, device=device)
        self.batch_size = batch_size
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.i_entail = labels["entailment"]
        self.i_contra = labels["contradiction"]

    def score(self, premises: Sequence[str], claim: str) -> list[Score]:
        return self.score_many(premises, [claim])[0]

    def score_many(self, premises: Sequence[str], claims: Sequence[str]) -> list[list[Score]]:
        """All (premise, claim) pairs of one answer in a single batched model call."""
        if not premises or not claims:
            return [[] for _ in claims]
        import numpy as np

        pairs = [(p, c) for c in claims for p in premises]
        logits = np.atleast_2d(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        flat = [Score(float(p[self.i_entail]), float(p[self.i_contra])) for p in probs]
        n = len(premises)
        return [flat[i * n:(i + 1) * n] for i in range(len(claims))]


class CosineScorer:
    """Embedding cosine similarity, mapped to 0..1. Can't see contradictions."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = f"cosine:{model.split('/')[-1]}"
        self.model = SentenceTransformer(model, device=device)

    def score(self, premises: Sequence[str], claim: str) -> list[Score]:
        return self.score_many(premises, [claim])[0]

    def score_many(self, premises: Sequence[str], claims: Sequence[str]) -> list[list[Score]]:
        """Encode the context windows once per answer, not once per sentence."""
        if not premises or not claims:
            return [[] for _ in claims]
        emb = self.model.encode([*claims, *premises], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        c, p = emb[: len(claims)], emb[len(claims):]
        sims = c @ p.T
        return [[Score(float(max(0.0, s))) for s in row] for row in sims]


_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = set(
    "a an the and or but if then of to in on at by for with from as is are was were be been being it its this that these "
    "those there their they them he she his her we our you your i not no can could will would should may might also "
    "which who whom whose what when where why how than more most such into about over after before between during "
    "based provided passages passage context according".split()
)


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class OverlapScorer:
    """Fraction of the claim's content words (incl. numbers) that appear in the premise."""

    name = "overlap"

    def score(self, premises: Sequence[str], claim: str) -> list[Score]:
        cw = content_words(claim)
        if not cw:
            return [Score(1.0) for _ in premises]
        return [Score(len(cw & content_words(p)) / len(cw)) for p in premises]


def make_scorer(kind: str, model: str | None = None) -> Scorer:
    if kind == "nli":
        return NLIScorer(model or "cross-encoder/nli-deberta-v3-base")
    if kind == "cosine":
        return CosineScorer(model or "sentence-transformers/all-MiniLM-L6-v2")
    if kind == "overlap":
        return OverlapScorer()
    raise ValueError(f"unknown scorer: {kind}")
