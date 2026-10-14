"""Pick HalluGuard's settings from data, on the TRAIN split only.

Tries context window sizes and whether to give each sentence its previous
sentence for context (pronouns), and reports AUROC for each. The test split
is never touched, so the final benchmark stays fair.

    python bench/sweep.py --data-dir <RAGTruth parquet folder> [--per-task 15]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from halluguard.bench import TASKS, auroc, load_ragtruth, sample  # noqa: E402
from halluguard.check import score_claims  # noqa: E402
from halluguard.scorers import make_scorer  # noqa: E402


def overlaps(a, spans):
    return any(a[0] < e and s < a[1] for s, e in spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--per-task", type=int, default=15)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    scorer = make_scorer("nli", a.model)
    data = sample(load_ragtruth("train", a.data_dir), a.per_task, a.seed)
    print(f"{len(data)} train answers ({a.per_task} per task), model {scorer.name}\n")
    configs = [(120, False), (120, True), (50, False), (50, True), (25, False), (25, True)]
    print(f"{'window':>7} {'prev':>5} {'resp AUROC':>11} {'sent AUROC':>11} {'time':>7}")
    for max_words, prev in configs:
        t0 = time.time()
        rt, rs, st, ss = [], [], [], []
        for e in data:
            claims = score_claims(e.answer, e.context, scorer, max_words=max_words, with_previous=prev)
            rt.append(e.hallucinated)
            rs.append(1 - min((c[1] for c in claims), default=1.0))
            for span, sup, _, _ in claims:
                st.append(overlaps((span.start, span.end), e.spans))
                ss.append(1 - sup)
        print(f"{max_words:>7} {str(prev):>5} {auroc(rt, rs):>11.3f} {auroc(st, ss):>11.3f} {time.time() - t0:>6.0f}s", flush=True)
    print("\nHigher AUROC is better. Paste this table back.")


if __name__ == "__main__":
    main()
