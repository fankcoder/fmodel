from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset[tuple[Tensor, Tensor]]):
    """Turns one long token stream into shifted fixed-length next-token examples."""

    def __init__(self, token_ids: list[int], context_length: int) -> None:
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        if len(self.tokens) <= context_length:
            raise ValueError("not enough tokens for one training example")

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x = self.tokens[index : index + self.context_length]
        y = self.tokens[index + 1 : index + self.context_length + 1]
        return x, y


def load_tokenizer(model_path: str | Path) -> spm.SentencePieceProcessor:
    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.load(str(model_path)):
        raise RuntimeError(f"could not load tokenizer: {model_path}")
    return tokenizer


def encode_text(path: str | Path, tokenizer: spm.SentencePieceProcessor) -> list[int]:
    text = Path(path).read_text(encoding="utf-8")
    return tokenizer.encode(text, out_type=int)
