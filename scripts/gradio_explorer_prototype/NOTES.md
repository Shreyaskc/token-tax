A working Gradio version of the explorer (heatmap + cost calculator),
functionally identical to `hf_space/index.html` but as a live Python app
instead of a static page. Not deployed: Hugging Face Spaces requires a PRO
subscription to host Gradio/Docker Spaces even on the free CPU tier — only
Static Spaces are free. `hf_space/` is the static rewrite that's actually
published.

If the account upgrades to PRO later, this is ready to go:

    python scripts/publish_hf_space.py --sdk gradio \
        --source scripts/gradio_explorer_prototype --repo-id shreyaskc/tokentax-explorer

Advantages this version has over the static page: reads
`shreyaskc/tokentax-results-v1` live via `hf://` at request time (always
current, no rebuild step) instead of a bundled JSON snapshot.
