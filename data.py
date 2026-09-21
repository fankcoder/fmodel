from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import numpy as np
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


class MemmapTokenBlockDataset(Dataset[tuple[Tensor, Tensor]]):
    """Reads non-overlapping next-token blocks from a uint16 token file on disk.

    The fixed block stride avoids both loading the corpus into RAM and creating a
    huge random permutation for every possible one-token offset.
    """

    def __init__(self, token_file: str | Path, context_length: int) -> None:
        self.path = Path(token_file)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        self.num_blocks = max(0, (len(self.tokens) - context_length - 1) // context_length + 1)
        if self.num_blocks == 0:
            raise ValueError("not enough tokens for one training block")

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.context_length
        # PyTorch 2.0 cannot construct a tensor from NumPy uint16 directly.
        x = torch.from_numpy(np.array(self.tokens[start : start + self.context_length], dtype=np.int64))
        y = torch.from_numpy(np.array(self.tokens[start + 1 : start + self.context_length + 1], dtype=np.int64))
        return x, y


def load_tokenizer(model_path: str | Path) -> spm.SentencePieceProcessor:
    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.load(str(model_path)):
        raise RuntimeError(f"could not load tokenizer: {model_path}")
    return tokenizer


def encode_text(path: str | Path, tokenizer: spm.SentencePieceProcessor) -> list[int]:
    text = Path(path).read_text(encoding="utf-8")
    return tokenizer.encode(text, out_type=int)
