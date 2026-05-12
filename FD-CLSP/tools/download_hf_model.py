from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


REPO_FILES = {
    "gpt2": [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "model.safetensors",
        "pytorch_model.bin",
    ],
    "sshleifer/tiny-gpt2": [
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "pytorch_model.bin",
    ],
}

OPTIONAL_FILES = {"special_tokens_map.json", "model.safetensors", "pytorch_model.bin"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = REPO_FILES.get(args.repo_id)
    if files is None:
        raise ValueError(f"Unsupported repo-id: {args.repo_id}")

    with urllib.request.urlopen(f"https://huggingface.co/api/models/{args.repo_id}", timeout=30) as response:
        model_info = json.load(response)
    available_files = {entry["rfilename"] for entry in model_info.get("siblings", [])}

    for filename in files:
        if filename not in available_files:
            if filename in OPTIONAL_FILES:
                print(f"skip missing {filename}")
                continue
            raise FileNotFoundError(f"{filename} not found in repo {args.repo_id}")
        url = f"https://huggingface.co/{args.repo_id}/resolve/main/{filename}"
        target = output_dir / filename
        if target.exists():
            print(f"skip {filename}")
            continue
        print(f"download {filename}")
        urllib.request.urlretrieve(url, target)


if __name__ == "__main__":
    main()
