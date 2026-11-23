---
title: HalluGuard
emoji: 🛡️
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Flags RAG answer sentences the source doesn't support (NLI)
---

# HalluGuard

Paste a source and an answer. Each sentence of the answer is checked against the source with an NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`) and marked supported, not in source, or contradicted.

Code and the RAGTruth benchmark: https://github.com/nancy6547/halluguard
