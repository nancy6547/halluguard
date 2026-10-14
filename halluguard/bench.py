"""Benchmark HalluGuard on RAGTruth (Niu et al., 2024).

RAGTruth has ~18k RAG answers from 6 LLMs across three tasks (QA, Data2txt,
Summary), with human-labelled hallucinated spans. We measure two things:

  response level  does the answer contain any hallucination?
  sentence level  which sentences contain one? (a sentence is positive if it
                  overlaps a labelled span)

The decision threshold is tuned on a sample of the TRAIN split, then frozen
and applied to a separate sample of the TEST split, so test numbers are not
tuned on test data. Scores are cached, so re-running is instant.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .check import score_claims
from .scorers import Scorer

TASKS = ("QA", "Data2txt", "Summary")


@dataclass
class Example:
    id: str
    task: str
    context: str
    answer: str
    spans: list[tuple[int, int]]  # labelled hallucinated character spans in `answer`

    @property
    def hallucinated(self) -> bool:
        return bool(self.spans)


def load_ragtruth(split: str, data_dir: str | None = None) -> list[Example]:
    """Load from local parquet files (`data_dir`) or the Hugging Face hub."""
    import pandas as pd

    if data_dir:
        files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {split}-*.parquet in {data_dir}")
        df = pd.concat(pd.read_parquet(f) for f in files)
    else:
        from huggingface_hub import hf_hub_download

        f = hf_hub_download("wandb/RAGTruth-processed", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(f)
    out = []
    for r in df.itertuples():
        if r.quality != "good":
            continue
        labels = json.loads(r.hallucination_labels) if r.hallucination_labels else []
        out.append(Example(str(r.id), r.task_type, r.context, r.output, [(int(l["start"]), int(l["end"])) for l in labels]))
    return out


def sample(examples: list[Example], per_task: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for t in TASKS:
        pool = [e for e in examples if e.task == t]
        out += rng.sample(pool, min(per_task, len(pool)))
    return out


def score_examples(examples: list[Example], scorer: Scorer, cache_file: Path, log=print) -> dict[str, list[dict]]:
    """Per example: list of claims with offsets, support and contradiction. Cached as JSONL."""
    cache: dict[str, list[dict]] = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["id"]] = row["claims"]
    todo = [e for e in examples if e.id not in cache]
    if todo:
        log(f"scoring {len(todo)} answers with {scorer.name} ({len(examples) - len(todo)} cached)")
    t0 = time.time()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("a", encoding="utf-8") as f:
        for n, e in enumerate(todo, 1):
            claims = [
                {"start": sp.start, "end": sp.end, "support": sup, "contradiction": con}
                for sp, sup, con, _ in score_claims(e.answer, e.context, scorer)
            ]
            cache[e.id] = claims
            f.write(json.dumps({"id": e.id, "claims": claims}) + "\n")
            if n % 25 == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-9)
                log(f"  {n}/{len(todo)}  ({rate:.1f} answers/s, ~{(len(todo) - n) / rate / 60:.1f} min left)")
    return cache


def _overlaps(a: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(a[0] < e and s < a[1] for s, e in spans)


def predictions(examples, scores, threshold):
    """(response_true, response_pred, sentence_true, sentence_pred, response_score) lists."""
    rt, rp, st, sp, rs = [], [], [], [], []
    for e in examples:
        claims = scores[e.id]
        flags = [c["support"] < threshold for c in claims]
        rt.append(e.hallucinated)
        rp.append(any(flags))
        rs.append(1 - min((c["support"] for c in claims), default=1.0))
        for c, flag in zip(claims, flags):
            st.append(_overlaps((c["start"], c["end"]), e.spans))
            sp.append(flag)
    return rt, rp, st, sp, rs


def prf(y_true, y_pred) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "n": len(y_true), "positives": sum(y_true)}


def auroc(y_true, scores) -> float:
    """Probability a random hallucinated answer scores higher than a random clean one."""
    pos = [s for t, s in zip(y_true, scores) if t]
    neg = [s for t, s in zip(y_true, scores) if not t]
    if not pos or not neg:
        return float("nan")
    ranked = sorted((s, t) for s, t in zip(scores, y_true))
    # average ranks for ties (Mann-Whitney U)
    ranks, i = {}, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    r_pos = sum(ranks[k] for k, (_, t) in enumerate(ranked) if t)
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def tune_threshold(examples, scores) -> float:
    """Threshold that maximises response-level F1 on the tuning set."""
    best_t, best_f = 0.5, -1.0
    for i in range(1, 100):
        t = i / 100
        rt, rp, *_ = predictions(examples, scores, t)
        f = prf(rt, rp)["f1"]
        if f > best_f:
            best_t, best_f = t, f
    return best_t


def evaluate(examples, scores, threshold) -> dict:
    out = {"threshold": threshold, "by_task": {}}
    rt, rp, st, sp, rs = predictions(examples, scores, threshold)
    out["response"] = prf(rt, rp) | {"auroc": auroc(rt, rs)}
    out["sentence"] = prf(st, sp)
    for t in TASKS:
        ex = [e for e in examples if e.task == t]
        a, b, c, d, s = predictions(ex, scores, threshold)
        out["by_task"][t] = {"response": prf(a, b) | {"auroc": auroc(a, s)}, "sentence": prf(c, d)}
    return out


def run(scorer: Scorer, per_task: int = 100, tune_per_task: int = 50, seed: int = 7, data_dir: str | None = None, out_dir: str = "results", log=print) -> dict:
    out = Path(out_dir)
    slug = scorer.name.replace(":", "_").replace("/", "_")
    train = sample(load_ragtruth("train", data_dir), tune_per_task, seed)
    test = sample(load_ragtruth("test", data_dir), per_task, seed)
    log(f"tuning on {len(train)} train answers, testing on {len(test)} test answers")
    s_train = score_examples(train, scorer, out / f"cache_{slug}_train.jsonl", log)
    threshold = tune_threshold(train, s_train)
    log(f"threshold tuned on train: {threshold:.2f}")
    s_test = score_examples(test, scorer, out / f"cache_{slug}_test.jsonl", log)
    rt = [e.hallucinated for e in test]
    log(f"for reference, flagging every answer gives response F1 {prf(rt, [True] * len(rt))['f1']:.3f}")
    res = {"scorer": scorer.name, "seed": seed, "per_task": per_task, "tune_per_task": tune_per_task} | evaluate(test, s_test, threshold)
    (out / f"bench_{slug}.json").write_text(json.dumps(res, indent=2))
    return res


def baseline_row(per_task: int, seed: int, data_dir: str | None) -> str:
    """'Flag every answer': the floor any detector has to beat."""
    test = sample(load_ragtruth("test", data_dir), per_task, seed)
    f = prf([e.hallucinated for e in test], [True] * len(test))["f1"]
    by = [prf([e.hallucinated for e in test if e.task == t], [True] * sum(e.task == t for e in test))["f1"] for t in TASKS]
    return f"| flag every answer | {f:.3f} | 0.500 | n/a | " + " | ".join(f"{x:.3f}" for x in by) + " |"


def table(results: list[dict], baseline: str | None = None) -> str:
    """Markdown table comparing scorers."""
    lines = [
        "| Scorer | Response F1 | Response AUROC | Sentence F1 | QA F1 | Data2txt F1 | Summary F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    if baseline:
        lines.append(baseline)
    for r in results:
        bt = r["by_task"]
        lines.append(
            f"| {r['scorer']} | {r['response']['f1']:.3f} | {r['response']['auroc']:.3f} | {r['sentence']['f1']:.3f} | "
            + " | ".join(f"{bt[t]['response']['f1']:.3f}" for t in TASKS)
            + " |"
        )
    return "\n".join(lines)
