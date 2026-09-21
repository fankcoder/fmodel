"""Encode a text corpus to memory-mappable uint16 Tiny GPT training files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/validation uint16 token files without loading the corpus into RAM.")
    parser.add_argument("--input", required=True, help="UTF-8 plain-text source corpus")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="artifacts/data")
    parser.add_argument("--validation-every", type=int, default=100, help="write every Nth source line to validation")
    parser.add_argument("--batch-lines", type=int, default=4096)
    args = parser.parse_args()
    if args.validation_every < 2:
        raise ValueError("validation-every must be at least 2")

    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = output_dir / "train.bin", output_dir / "valid.bin"
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.get_piece_size() > np.iinfo(np.uint16).max:
        raise ValueError("uint16 storage needs a vocabulary of at most 65,535 tokens")
    eos_id = tokenizer.eos_id()
    train_tokens = valid_tokens = source_lines = 0

    with source.open("r", encoding="utf-8") as text, train_path.open("wb") as train_out, valid_path.open("wb") as valid_out:
        lines: list[str] = []
        progress = tqdm(desc="Encoding lines", unit=" lines")
        for line in text:
            lines.append(line.rstrip("\n"))
            if len(lines) < args.batch_lines:
                continue
            encoded = tokenizer.encode(lines, out_type=int)
            for sentence in encoded:
                sentence.append(eos_id)
                target = valid_out if source_lines % args.validation_every == 0 else train_out
                np.asarray(sentence, dtype=np.uint16).tofile(target)
                if source_lines % args.validation_every == 0:
                    valid_tokens += len(sentence)
                else:
                    train_tokens += len(sentence)
                source_lines += 1
            progress.update(len(lines))
            lines.clear()
        if lines:
            encoded = tokenizer.encode(lines, out_type=int)
            for sentence in encoded:
                sentence.append(eos_id)
                target = valid_out if source_lines % args.validation_every == 0 else train_out
                np.asarray(sentence, dtype=np.uint16).tofile(target)
                if source_lines % args.validation_every == 0:
                    valid_tokens += len(sentence)
                else:
                    train_tokens += len(sentence)
                source_lines += 1
            progress.update(len(lines))
        progress.close()
    print(f"Prepared {source_lines:,} lines")
    print(f"Train: {train_path} ({train_tokens:,} tokens, {train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Valid: {valid_path} ({valid_tokens:,} tokens, {valid_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
