---
title: tokentax Explorer
emoji: 🌍
colorFrom: red
colorTo: yellow
sdk: static
pinned: false
license: cc0-1.0
---

# tokentax explorer

Interactive heatmap and cost calculator for the tokenizer cost penalty
across 203 FLORES-200 languages and 8 tokenizers.

- Data: [`shreyaskc/tokentax-results-v1`](https://huggingface.co/datasets/shreyaskc/tokentax-results-v1)
- Code: [Shreyaskc/token-tax](https://github.com/Shreyaskc/token-tax)

A static page (Plotly.js + vanilla JS) — Hugging Face Spaces requires a PRO
subscription to host Gradio/Docker apps even on the free CPU tier, so
`data.json`/`pricing.json` here are a **snapshot as of build time**, not a
live Hub read. Regenerate with `scripts/build_static_explorer.py` in the
source repo after publishing a new results dataset version, then
`scripts/publish_hf_space.py` to redeploy. A live Gradio version exists at
`scripts/gradio_explorer_prototype/` for if the account upgrades to PRO.
