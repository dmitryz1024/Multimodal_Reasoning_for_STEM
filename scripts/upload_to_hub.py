"""
Upload a trained checkpoint folder to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a trained checkpoint to the Hugging Face Hub")
    parser.add_argument("--local_path", type=str, required=True, help="Path to the local checkpoint folder")
    parser.add_argument("--repo_id", type=str, required=True, help="Target Hub repo, e.g. username/model-name")
    parser.add_argument("--private", action="store_true", help="Create the repository as private")
    parser.add_argument("--commit_message", type=str, default="Upload trained checkpoint", help="Commit message")
    args = parser.parse_args()

    load_dotenv()

    local_path = Path(args.local_path).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {local_path}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(local_path),
        commit_message=args.commit_message,
    )

    print(f"Uploaded checkpoint to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
