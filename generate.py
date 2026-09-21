from __future__ import annotations

import argparse
import sys
import torch

from data import load_tokenizer
from model import TinyGPT, TinyGPTConfig


def main() -> None:
    # SentencePiece may emit U+2047 for an unknown token; Windows terminals are
    # often configured for GBK, which cannot display it without this setting.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate text with a trained Tiny GPT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TinyGPT(TinyGPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = load_tokenizer(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt, out_type=int)
    if not prompt_ids:
        raise ValueError("prompt produced no tokens")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, args.max_new_tokens, args.temperature, args.top_k)[0].tolist()
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
