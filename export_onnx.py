"""Export a trained Tiny GPT checkpoint to ONNX for the deployment stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from model import TinyGPT, TinyGPTConfig


class LogitsOnly(nn.Module):
    def __init__(self, model: TinyGPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, token_ids: Tensor) -> Tensor:
        logits, _ = self.model(token_ids)
        return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Tiny GPT logits to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/tinygpt.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = TinyGPT(TinyGPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    example = torch.zeros((1, model.config.context_length), dtype=torch.long)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        LogitsOnly(model), example, str(destination), input_names=["token_ids"], output_names=["logits"],
        dynamic_axes={"token_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        opset_version=args.opset,
    )
    print(f"Saved ONNX model to {destination}")


if __name__ == "__main__":
    main()
