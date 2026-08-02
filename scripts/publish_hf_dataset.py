"""Phase 3: publish the Phase 2 sweep results as a Hugging Face dataset.

Converts the local CSV/parquet results into the dataset's published parquet
files, then creates (if needed) and uploads to a Hugging Face dataset repo.
Re-running with the same --repo-id re-uploads (new commit on the HF repo);
pass a new --repo-id (e.g. tokentax-results-v2) for a new version rather than
overwriting v1's history.

Requires `huggingface-cli login` (already needed for the sweep itself) with
write access to the target repo/namespace.

Usage:
    python scripts/publish_hf_dataset.py [--repo-id shreyaskc/tokentax-results-v1] [--private]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
HF_DATASET_DIR = ROOT / "hf_dataset"


def build_release_files(staging_dir: Path):
    summary_csv = RESULTS_DIR / "tokentax_summary_devtest.csv"
    raw_parquet = RESULTS_DIR / "tokentax_raw_devtest.parquet"
    domain_csv = RESULTS_DIR / "opus_domain_check.csv"

    for path in (summary_csv, raw_parquet, domain_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run run_flores_sweep.py, summarize_with_evalci.py, "
                "and run_opus_domain_check.py first"
            )

    pd.read_csv(summary_csv).to_parquet(staging_dir / "flores200_summary.parquet")
    shutil.copy(raw_parquet, staging_dir / "flores200_raw.parquet")
    pd.read_csv(domain_csv).to_parquet(staging_dir / "opus_domain_check.parquet")
    shutil.copy(HF_DATASET_DIR / "DATASET_CARD.md", staging_dir / "README.md")
    shutil.copy(HF_DATASET_DIR / "LICENSE", staging_dir / "LICENSE")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="shreyaskc/tokentax-results-v1")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp:
        staging_dir = Path(tmp)
        build_release_files(staging_dir)

        print(f"Creating/reusing dataset repo {args.repo_id}...")
        api.create_repo(
            repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True
        )

        print(f"Uploading {list(staging_dir.iterdir())} to {args.repo_id}...")
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Publish tokentax results v1",
        )

    print(f"\nPublished: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
