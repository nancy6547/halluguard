# HalluGuard

[![test](https://github.com/nancy6547/halluguard/actions/workflows/test.yml/badge.svg)](https://github.com/nancy6547/halluguard/actions/workflows/test.yml)

**Flag the sentences in a RAG answer that its sources don't support.**

A RAG app retrieves the right document and the LLM still adds a number, a date or a name that isn't in it. HalluGuard splits the answer into sentences and checks each one against every part of the retrieved context with a natural language inference (NLI) model. Every sentence comes back as **supported**, **unsupported** (not in the source) or **contradicted** (the source says otherwise), along with the passage that backs it up.

```
$ halluguard check --context examples/eiffel_context.txt --answer examples/eiffel_answer.txt
OK  <support>  The Eiffel Tower in Paris was completed in 1889 for the World's Fair.
OK  <support>  It is 330 metres tall.
XX  <support>  It was built from steel to resist corrosion.
??  <support>  Around 7 million people visit each year, and it was the tallest structure in Europe until 1990.

HALLUCINATION SUSPECTED (2/4 flagged)
```

`OK` supported · `??` not in the source · `XX` contradicts the source. Each line shows the sentence's best support score (0 to 1).

## Why not cosine similarity?

My first version scored each sentence by embedding similarity to the source. It failed in exactly the cases that matter: a wrong answer that reuses the source's words scores *high*.

| Source says | Answer says | Cosine | NLI |
|---|---|---|---|
| "The tower is made of wrought iron." | "The tower is made of steel." | high (same topic, same words) | **contradiction** |
| "Revenue was $4.2M in 2023." | "Revenue was $9.1M in 2021." | high | **not entailed** |

Similarity measures *what a sentence is about*. NLI measures *whether the source implies it*, which is the question a hallucination detector has to answer. HalluGuard uses `cross-encoder/nli-deberta-v3-base` by default.

## Results on RAGTruth

[RAGTruth](https://arxiv.org/abs/2401.00396) is a benchmark of about 18,000 answers from six LLMs across three RAG tasks (question answering, data-to-text, summarisation), where people labelled every hallucinated span.

**Setup.** 100 test answers per task (300 total). The threshold for each scorer is tuned on a *separate* sample from the train split (50 per task), then frozen, so the test numbers aren't tuned on test data. Sentence-level: a sentence counts as hallucinated if it overlaps a labelled span. Seed 7, so anyone can reproduce it with one command.

<!-- BENCH:START -->
| Scorer | Response F1 | Response AUROC | Sentence F1 | QA F1 | Data2txt F1 | Summary F1 |
|---|---|---|---|---|---|---|
| flag every answer | 0.544 | 0.500 | n/a | 0.319 | 0.824 | 0.374 |
| overlap | 0.611 | 0.747 | 0.228 | 0.410 | 0.824 | 0.412 |
<!-- BENCH:END -->

- **flag every answer** is the floor: any detector must beat it. Data-to-text answers are hallucinated about 64% of the time in RAGTruth, so "always flag" already scores well there.
- **overlap** is a no-model baseline: the share of a sentence's content words (including numbers) found in the source.
- AUROC is threshold-free: the chance that a random hallucinated answer gets a worse score than a random clean one.

Reproduce:

```bash
pip install -e ".[dev]"
halluguard bench --scorer overlap cosine nli    # downloads RAGTruth + models on first run
python bench/update_readme.py                   # writes the table above
```

Scores are cached in `results/`, so re-running only scores what's new.

## How it works

```
answer ──► split into sentences (claims), keep character offsets
context ─► overlapping windows of whole sentences (≤120 words; NLI reads ≤512 tokens)

for each claim:
    score (window, claim) with the NLI cross-encoder, for every window
    support       = best entailment probability over all windows
    contradiction = highest contradiction probability
    verdict       = supported     if support ≥ threshold
                    contradicted  if contradiction ≥ 0.5 and > support
                    unsupported   otherwise
answer is flagged if any claim is not supported
```

- **Windows overlap by half**, so a fact that sits across a boundary is still read in one piece.
- **Sentences under 4 words are skipped** ("Yes.", "In summary,"); they rarely carry checkable facts.
- **Offsets are kept**, so a UI can highlight exactly which sentence failed (see the demo and [DocMind](https://github.com/nancy6547/docmind)).

## Use it

```python
from halluguard import NLIScorer, check

rep = check(answer, context, NLIScorer(), threshold=0.5)
for c in rep.claims:
    print(c.verdict, round(c.support, 2), c.text)
rep.hallucinated   # True if any sentence is unsupported or contradicted
```

**Demo app:** `pip install -e ".[app]"` then `streamlit run app.py`. It highlights each sentence green, amber or red and shows the passage it was checked against.

## Limitations

- **Sentence = claim.** A sentence with two facts, one right and one wrong, is judged as one unit. Splitting into atomic claims with an LLM would be finer but slower and non-deterministic.
- **Paraphrase vs. inference.** NLI handles paraphrase well, but multi-hop reasoning across distant parts of a long document can be missed when the two facts land in different windows.
- **Speed.** Every sentence is checked against every window, so long contexts cost more. A smaller model (`--model cross-encoder/nli-deberta-v3-xsmall`) is faster at some cost in accuracy.
- English only.

## Project structure

```
halluguard/
  split.py     sentences with offsets, overlapping context windows
  scorers.py   NLI, cosine and word-overlap scorers (same interface)
  check.py     per-claim verdicts and the answer-level report
  bench.py     RAGTruth loading, threshold tuning on train, metrics (F1, AUROC)
  cli.py       `halluguard check` and `halluguard bench`
app.py         Streamlit demo
tests/         unit tests (no model download needed)
```

---

Built by **Nancy Singh**: [GitHub](https://github.com/nancy6547) · [LinkedIn](https://linkedin.com/in/nancy-singh-84367722a) · [Portfolio](https://nancy6547.github.io)
