"""Phase 3: publish the interactive explorer as a Hugging Face Space.

Uploads hf_space/ (index.html, data.json, pricing.json, README.md with Space
metadata) to a Static Space — free on any HF account, unlike Gradio/Docker
Spaces which require PRO even on the CPU-basic tier. data.json/pricing.json
are a snapshot as of the last scripts/build_static_explorer.py run, not a
live Hub read; re-run that script before this one after publishing a new
results dataset version.

Requires `huggingface-cli login` with write access to the target namespace.

Usage:
    python scripts/publish_hf_space.py [--repo-id shreyaskc/tokentax-explorer] [--private]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_SPACE_DIR = ROOT / "hf_space"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="shreyaskc/tokentax-explorer")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()

    print(f"Creating/reusing Space repo {args.repo_id}...")
    api.create_repo(
        repo_id=args.repo_id, repo_type="space", space_sdk="static",
        private=args.private, exist_ok=True,
    )

    print(f"Uploading {HF_SPACE_DIR} to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(HF_SPACE_DIR),
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Publish tokentax explorer",
    )

    print(f"\nPublished: https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
