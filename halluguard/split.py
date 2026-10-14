"""Split text into sentences ("claims") and keep their character offsets.

A small rule-based splitter, so HalluGuard has no extra dependencies and the
offsets line up exactly with the original answer (needed to compare against
RAGTruth's labelled spans).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Words that end with a period but don't end a sentence.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co", "corp",
    "e.g", "i.e", "u.s", "u.k", "no", "fig", "approx", "dept", "est", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "a.m", "p.m",
}
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int


def sentences(text: str, min_words: int = 1) -> list[Span]:
    """Return sentences with (start, end) offsets into `text`."""
    out: list[Span] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        end = m.end()
        chunk = text[start:end]
        if not chunk.strip():
            start = end
            continue
        # Don't split after an abbreviation ("Dr. Smith") or inside a number ("3.5").
        before = text[start:m.start()].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if "\n" not in m.group() and (last in _ABBREV or (len(last) == 1 and last.isalpha())):
            continue
        out.append(_trim(text, start, end))
        start = end
    if text[start:].strip():
        out.append(_trim(text, start, len(text)))
    return [s for s in out if len(s.text.split()) >= min_words]


def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(text[start:end], start, end)


def windows(text: str, max_words: int = 120) -> list[str]:
    """Overlapping windows of whole sentences, each at most ~max_words long.

    NLI models read at most ~512 tokens, so a long context is checked piece by
    piece; overlap keeps facts that straddle a boundary in one window.
    """
    sents = [s.text for s in sentences(text)]
    if not sents:
        return [text] if text.strip() else []
    out: list[str] = []
    i = 0
    while i < len(sents):
        words, j = 0, i
        while j < len(sents) and (words == 0 or words + len(sents[j].split()) <= max_words):
            words += len(sents[j].split())
            j += 1
        out.append(" ".join(sents[i:j]))
        if j >= len(sents):
            break
        i += max(1, (j - i) // 2)  # next window starts halfway through this one
    return out
