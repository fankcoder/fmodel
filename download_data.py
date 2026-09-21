"""Download the official TinyStories plain-text train and validation files."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
FILES = {"tinystories.txt": "TinyStories-train.txt", "tinystories_valid.txt": "TinyStories-valid.txt"}


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Keeping existing {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except OSError as error:
        raise RuntimeError(
            f"Could not download {url}. Check access to huggingface.co, or place the official "
            f"file at {destination} manually."
        ) from error
    temporary.replace(destination)
    print(f"Saved {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for local_name, remote_name in FILES.items():
        download(f"{BASE_URL}/{remote_name}", output_dir / local_name)


if __name__ == "__main__":
    main()
