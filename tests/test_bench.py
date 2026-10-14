from halluguard.bench import Example, auroc, evaluate, predictions, prf, tune_threshold


def ex(i, answer, spans):
    return Example(str(i), "QA", "ctx", answer, spans)


def test_prf():
    assert prf([True, True, False, False], [True, False, True, False]) == {
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "n": 4, "positives": 2,
    }


def test_auroc_perfect_and_random():
    assert auroc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc([True, False], [0.5, 0.5]) == 0.5


def test_sentence_labels_use_span_overlap():
    e = ex(1, "Good sentence here. Bad sentence here.", [(20, 38)])
    scores = {"1": [{"start": 0, "end": 19, "support": 0.9}, {"start": 20, "end": 38, "support": 0.1}]}
    rt, rp, st, sp, _ = predictions([e], scores, 0.5)
    assert (rt, rp) == ([True], [True])
    assert st == [False, True] and sp == [False, True]


def test_threshold_tuning_separates_classes():
    exs = [ex(1, "a", [(0, 1)]), ex(2, "b", [])]
    scores = {"1": [{"start": 0, "end": 1, "support": 0.2}], "2": [{"start": 0, "end": 1, "support": 0.8}]}
    t = tune_threshold(exs, scores)
    assert 0.2 < t <= 0.8
    assert evaluate(exs, scores, t)["response"]["f1"] == 1.0
