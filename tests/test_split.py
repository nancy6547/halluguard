from halluguard.split import sentences, windows


def test_offsets_match_text():
    t = "Dr. Smith paid $3.50 at 5 p.m. on Monday. It rained!\nNew line here. The U.S. team won 2.5 points"
    ss = sentences(t)
    assert [s.text for s in ss] == [
        "Dr. Smith paid $3.50 at 5 p.m. on Monday.",
        "It rained!",
        "New line here.",
        "The U.S. team won 2.5 points",
    ]
    assert all(t[s.start:s.end] == s.text for s in ss)


def test_min_words_drops_fragments():
    assert [s.text for s in sentences("Yes. The tower is 330 metres tall.", min_words=4)] == ["The tower is 330 metres tall."]


def test_windows_cover_every_sentence_and_overlap():
    text = " ".join(f"Fact number {i} is here." for i in range(40))
    ws = windows(text, max_words=25)
    assert all(len(w.split()) <= 25 for w in ws)
    for i in range(40):
        assert any(f"Fact number {i} is" in w for w in ws)
    assert len(ws) > 1 and ws[0].split(". ")[-1] in ws[1]


def test_windows_short_context_is_one_window():
    assert windows("Only one short sentence.") == ["Only one short sentence."]
