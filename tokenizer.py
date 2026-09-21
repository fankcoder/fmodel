from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for Tiny GPT.")
    parser.add_argument("--input", required=True, help="UTF-8 training text")
    parser.add_argument("--output-dir", default="artifacts/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000)
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "tinystories"
    spm.SentencePieceTrainer.train(
        input=str(source), model_prefix=str(prefix), vocab_size=args.vocab_size,
        model_type="bpe", character_coverage=1.0, normalization_rule_name="nmt_nfkc",
        bos_id=1, eos_id=2, unk_id=0, pad_id=3, hard_vocab_limit=False,
    )
    print(f"Saved tokenizer to {prefix}.model and {prefix}.vocab")


if __name__ == "__main__":
    main()
