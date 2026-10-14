"""HalluGuard: flag the sentences in a RAG answer that its sources don't support."""
from .check import CONTRADICTED, SUPPORTED, UNSUPPORTED, ClaimResult, Report, check, score_claims
from .scorers import CosineScorer, NLIScorer, OverlapScorer, Score, make_scorer
from .split import sentences, windows

__all__ = [
    "check", "score_claims", "Report", "ClaimResult", "SUPPORTED", "UNSUPPORTED", "CONTRADICTED",
    "NLIScorer", "CosineScorer", "OverlapScorer", "Score", "make_scorer", "sentences", "windows",
]
__version__ = "0.2.0"
