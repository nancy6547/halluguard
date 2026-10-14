"""halluguard check --context ctx.txt --answer answer.txt [--scorer nli] [--json]
halluguard bench [--scorer nli cosine overlap] [--per-task 100] [--data-dir path]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .check import CONTRADICTED, SUPPORTED, check
from .scorers import make_scorer

DEFAULT_THRESHOLD = {"nli": 0.5, "cosine": 0.6, "overlap": 0.7}


def _read(v: str) -> str:
    p = Path(v)
    return p.read_text(encoding="utf-8") if p.exists() else v


def main(argv=None):
    ap = argparse.ArgumentParser(prog="halluguard", description="Flag RAG answer sentences their sources don't support.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="check one answer against its context")
    c.add_argument("--context", required=True, help="text or path to a file")
    c.add_argument("--answer", required=True, help="text or path to a file")
    c.add_argument("--scorer", default="nli", choices=["nli", "cosine", "overlap"])
    c.add_argument("--model", default=None)
    c.add_argument("--threshold", type=float, default=None)
    c.add_argument("--json", action="store_true")

    b = sub.add_parser("bench", help="benchmark on RAGTruth")
    b.add_argument("--scorer", nargs="+", default=["overlap", "cosine", "nli"])
    b.add_argument("--per-task", type=int, default=100, help="test answers per task (QA, Data2txt, Summary)")
    b.add_argument("--tune-per-task", type=int, default=50, help="train answers per task used to pick the threshold")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--data-dir", default=None, help="folder with RAGTruth-processed parquet files (default: download)")
    b.add_argument("--nli-model", default=None, help="e.g. cross-encoder/nli-deberta-v3-xsmall (faster)")
    b.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    if a.cmd == "check":
        scorer = make_scorer(a.scorer, a.model)
        t = a.threshold if a.threshold is not None else DEFAULT_THRESHOLD[a.scorer]
        rep = check(_read(a.answer), _read(a.context), scorer, threshold=t)
        if a.json:
            print(json.dumps(rep.to_dict(), indent=2))
            return
        mark = {SUPPORTED: "OK ", CONTRADICTED: "XX ", "unsupported": "?? "}
        for cl in rep.claims:
            print(f"{mark[cl.verdict]} {cl.support:.2f}  {cl.text}")
        print(f"\n{'HALLUCINATION SUSPECTED' if rep.hallucinated else 'all claims supported'} ({sum(c.verdict != SUPPORTED for c in rep.claims)}/{len(rep.claims)} flagged)")
        sys.exit(1 if rep.hallucinated else 0)

    from .bench import baseline_row, run, table

    results = []
    for kind in a.scorer:
        print(f"\n== {kind} ==")
        results.append(run(make_scorer(kind, a.nli_model if kind == "nli" else None), a.per_task, a.tune_per_task, a.seed, a.data_dir, a.out))
    md = table(results, baseline_row(a.per_task, a.seed, a.data_dir))
    Path(a.out, "bench.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
