from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import MemmapTokenBlockDataset, TokenBlockDataset, encode_text, load_tokenizer
from model import TinyGPT, TinyGPTConfig, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny GPT from scratch.")
    parser.add_argument("--train-text")
    parser.add_argument("--valid-text", default=None)
    parser.add_argument("--train-bin", help="prepared uint16 train token file")
    parser.add_argument("--valid-bin", help="prepared uint16 validation token file")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default=None, help="checkpoint path, or 'last'")
    parser.add_argument("--no-amp", action="store_true", help="disable FP16 automatic mixed precision")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate_at(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup followed by cosine decay to 10% of the initial learning rate."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    return base_lr * (0.1 + 0.9 * cosine)


@torch.no_grad()
def evaluate(model: TinyGPT, loader: DataLoader, device: torch.device, amp_enabled: bool, batches: int) -> float:
    model.eval()
    losses: list[float] = []
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    for index, (x, y) in enumerate(loader):
        if index >= batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled) if device.type == "cuda" else autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path: Path, model: TinyGPT, optimizer: AdamW, scaler: torch.cuda.amp.GradScaler, step: int) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step, "config": asdict(model.config)}, path)


def save_inference_model(path: Path, model: TinyGPT) -> None:
    """Save only the weights and config, without AdamW state, for generation/export."""
    torch.save({"model": model.state_dict(), "config": asdict(model.config)}, path)


def main() -> None:
    args = parse_args()
    if bool(args.train_text) == bool(args.train_bin):
        raise ValueError("provide exactly one of --train-text or --train-bin")
    if args.train_bin and not args.valid_bin:
        raise ValueError("--valid-bin is required with --train-bin")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)
    config = TinyGPTConfig(vocab_size=tokenizer.get_piece_size())
    if config.vocab_size != 8000:
        print(f"Warning: tokenizer vocab is {config.vocab_size}; expected 8000 for the ~14M configuration.")

    if args.train_bin:
        train_data = MemmapTokenBlockDataset(args.train_bin, config.context_length)
        valid_data = MemmapTokenBlockDataset(args.valid_bin, config.context_length)
        train_token_count = len(train_data.tokens)
        valid_token_count = len(valid_data.tokens)
    else:
        train_tokens = encode_text(args.train_text, tokenizer)
        if args.valid_text:
            valid_tokens = encode_text(args.valid_text, tokenizer)
        else:
            split_at = int(len(train_tokens) * 0.99)
            train_tokens, valid_tokens = train_tokens[:split_at], train_tokens[split_at:]
        train_data = TokenBlockDataset(train_tokens, config.context_length)
        valid_data = TokenBlockDataset(valid_tokens, config.context_length)
        train_token_count, valid_token_count = len(train_tokens), len(valid_tokens)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    model = TinyGPT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_step = 0
    resume_path = output_dir / "last.pt" if args.resume == "last" else Path(args.resume) if args.resume else None
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"] + 1
        print(f"Resumed from {resume_path} at step {start_step}")

    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    print(f"Device: {device}; parameters: {count_parameters(model):,}; AMP: {amp_enabled}")
    print(f"Train tokens: {train_token_count:,}; validation tokens: {valid_token_count:,}")
    train_iter = iter(train_loader)
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    progress = tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps)
    model.train()
    final_step = start_step - 1
    for step in progress:
        lr = learning_rate_at(step, args.learning_rate, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(enabled=amp_enabled) if device.type == "cuda" else autocast():
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            total_loss += loss.item()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        progress.set_postfix(loss=f"{total_loss:.4f}", lr=f"{lr:.2e}", grad=f"{grad_norm:.2f}")

        if (step + 1) % args.eval_interval == 0:
            validation_loss = evaluate(model, valid_loader, device, amp_enabled, args.eval_batches)
            print(f"\nstep {step + 1}: train_loss={total_loss:.4f} val_loss={validation_loss:.4f}")
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(output_dir / f"step_{step + 1}.pt", model, optimizer, scaler, step)
            save_checkpoint(output_dir / "last.pt", model, optimizer, scaler, step)
        final_step = step
    if final_step >= start_step:
        save_checkpoint(output_dir / "last.pt", model, optimizer, scaler, final_step)
        save_inference_model(output_dir / "model.pt", model)


if __name__ == "__main__":
    main()
