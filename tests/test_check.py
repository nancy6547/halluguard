from halluguard import CONTRADICTED, SUPPORTED, UNSUPPORTED, OverlapScorer, Score, check


class FakeNLI:
    """Supports a claim if the premise contains it verbatim; contradicts if it contains 'NOT <claim>'."""

    name = "fake"

    def score(self, premises, claim):
        c = claim.rstrip(".")
        return [Score(1.0 if c in p and f"NOT {c}" not in p else 0.0, 0.9 if f"NOT {c}" in p else 0.0) for p in premises]


CTX = "The court dismissed the appeal. The evidence was found insufficient. NOT A fine was imposed on the company."


def test_verdict_per_claim():
    ans = "The court dismissed the appeal. A fine was imposed on the company. The judge was on holiday that week."
    rep = check(ans, CTX, FakeNLI(), threshold=0.5)
    assert [c.verdict for c in rep.claims] == [SUPPORTED, CONTRADICTED, UNSUPPORTED]
    assert rep.hallucinated
    assert rep.claims[0].evidence.startswith("The court dismissed")


def test_clean_answer_passes():
    rep = check("The court dismissed the appeal.", CTX, FakeNLI())
    assert not rep.hallucinated and rep.score == 1.0


def test_overlap_scorer_catches_new_numbers():
    s = OverlapScorer()
    ok, bad = s.score(["Revenue was 4.2 million in 2023."], "Revenue reached 4.2 million in 2023.")[0], s.score(
        ["Revenue was 4.2 million in 2023."], "Revenue reached 9.1 million in 2021."
    )[0]
    assert ok.support > bad.support
