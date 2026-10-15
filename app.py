"""HalluGuard demo: paste a source and an answer, see which sentences the source supports.

    pip install -e ".[app]"
    streamlit run app.py
"""
import html
import os

import streamlit as st

from halluguard import CONTRADICTED, SUPPORTED, NLIScorer, check

# The hosted demo uses the small model so it fits in Streamlit Cloud's memory.
# Run locally with HALLUGUARD_MODEL=cross-encoder/nli-deberta-v3-base for the benchmarked one.
MODEL = os.getenv("HALLUGUARD_MODEL", "cross-encoder/nli-deberta-v3-xsmall")

EXAMPLE_CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris. It was completed in 1889 as the entrance arch "
    "for the World's Fair. The tower is 330 metres tall, including its antennas. Its design is credited to the "
    "engineering company of Gustave Eiffel. About 7 million people visit it every year."
)
EXAMPLES = {}
EXAMPLE_ANSWER = (
    "The Eiffel Tower in Paris was completed in 1889 for the World's Fair. It is 330 metres tall. "
    "It was built from steel to resist corrosion. Around 7 million people visit each year, "
    "and it was the tallest structure in Europe until 1990."
)


EXAMPLES["Eiffel Tower (one wrong fact, one made up)"] = (EXAMPLE_CONTEXT, EXAMPLE_ANSWER)
EXAMPLES["Company report (numbers changed)"] = (
    "Northwind's revenue was $4.2 million in 2023, up from $3.1 million in 2022. The company has 340 employees "
    "and its headquarters is in Pune. It opened an office in Rotterdam in 2021.",
    "Northwind's revenue was $9.1 million in 2021. The company has 340 employees. "
    "Its headquarters is in Pune, and it opened a Rotterdam office in 2021.",
)
EXAMPLES["Leave policy (all supported)"] = (
    "Employees get 24 days of paid annual leave. Up to 5 unused leave days can be carried over to the next year. "
    "Sick leave longer than 3 consecutive days needs a doctor's note.",
    "You get 24 days of paid annual leave, and you can carry up to 5 unused days into the next year. "
    "For sick leave of more than 3 days in a row, you need a doctor's note.",
)


@st.cache_resource
def scorer():
    return NLIScorer(MODEL)


st.set_page_config(page_title="HalluGuard", page_icon="🛡️", layout="wide")
st.title("HalluGuard")
st.caption("Which sentences in this answer does the source actually support? Each sentence is checked with an NLI model against every part of the source.")

st.caption(
    f"Model: `{MODEL}` (the small one, so the free hosting can run it). "
    "[Code and RAGTruth benchmark on GitHub](https://github.com/nancy6547/halluguard)"
)
cols = st.columns(len(EXAMPLES))
for col, (name, (example_ctx, example_ans)) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.ctx, st.session_state.ans = example_ctx, example_ans

c1, c2 = st.columns(2)
ctx = c1.text_area("Source (the retrieved context)", key="ctx", height=260)
ans = c2.text_area("Answer (what the LLM said)", key="ans", height=260)
threshold = st.slider("Support needed to count as supported", 0.05, 0.95, 0.5, 0.05)

if st.button("Check answer", type="primary", disabled=not (ctx.strip() and ans.strip())):
    with st.spinner(f"Checking with {MODEL.split('/')[-1]}..."):
        rep = check(ans, ctx, scorer(), threshold=threshold)

    colors = {SUPPORTED: "#1d8a4f", CONTRADICTED: "#d62f2f"}
    parts, last = [], 0
    for c in rep.claims:
        parts.append(html.escape(ans[last:c.start]))
        color = colors.get(c.verdict, "#c27c0e")
        parts.append(
            f'<span title="{c.verdict}, support {c.support:.2f}" style="background:{color}22;'
            f'border-bottom:2px solid {color};padding:1px 2px">{html.escape(c.text)}</span>'
        )
        last = c.end
    parts.append(html.escape(ans[last:]))
    flagged = sum(c.verdict != SUPPORTED for c in rep.claims)
    st.subheader(f"{flagged} of {len(rep.claims)} sentences flagged" if flagged else "Every sentence is supported")
    st.markdown(f'<div style="font-size:1.05rem;line-height:1.9">{"".join(parts)}</div>', unsafe_allow_html=True)
    st.caption("Green: supported · Amber: not in the source · Red: contradicts the source. Hover a sentence for its score.")

    for c in rep.claims:
        with st.expander(f"{c.verdict.upper()} · support {c.support:.2f} · {c.text[:80]}"):
            st.write("**Best matching part of the source:**")
            st.write(c.evidence or "(none)")
            st.write(f"Contradiction score: {c.contradiction:.2f}")
