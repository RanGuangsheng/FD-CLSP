from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from fd_clip.data import CWRUClipDataset, build_grouped_splits
from fd_clip.model import FaultClipModel, class_contrastive_loss, prompt_alignment_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault-root", type=str, required=True)
    parser.add_argument("--normal-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/run1")
    parser.add_argument("--text-model", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-text-len", type=int, default=24)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--limit-per-file", type=int, default=None)
    parser.add_argument("--freeze-text-backbone", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--prompt-layers", type=int, default=1)
    parser.add_argument("--prompt-loss-weight", type=float, default=0.01)
    parser.add_argument("--prompt-loss-warmup-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_collate_fn(tokenizer, max_text_len: int):
    def collate_fn(batch):
        signals = torch.stack([item["signal"] for item in batch], dim=0)
        label_indices = torch.tensor([item["label_idx"] for item in batch], dtype=torch.long)
        return {
            "signals": signals,
            "label_indices": label_indices,
            "source_paths": [item["source_path"] for item in batch],
        }

    return collate_fn


@torch.no_grad()
def evaluate(model, data_loader, class_input_ids, class_attention_mask, device, num_classes: int) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        text_features = model.encode_text(class_input_ids, class_attention_mask)
        scale = model.logit_scale.exp().clamp(max=100.0)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in data_loader:
        signals = batch["signals"].to(device)
        label_indices = batch["label_indices"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            signal_features = model.encode_signal(signals)
            loss, logits = class_contrastive_loss(signal_features, text_features, label_indices, scale)

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == label_indices).sum().item()
        for target, pred in zip(label_indices.cpu(), predictions.cpu()):
            confusion[target.long(), pred.long()] += 1
        total_loss += loss.item() * signals.size(0)
        total += signals.size(0)

    class_accuracy = {}
    for class_idx in range(num_classes):
        class_total = confusion[class_idx].sum().item()
        class_accuracy[str(class_idx)] = 0.0 if class_total == 0 else confusion[class_idx, class_idx].item() / class_total

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "confusion_matrix": confusion.tolist(),
        "class_accuracy": class_accuracy,
    }


def summarize_split(dataset: CWRUClipDataset, subset) -> dict[str, object]:
    file_counts = defaultdict(set)
    sample_counts = defaultdict(int)
    for index in subset.indices:
        meta = dataset.sample_metas[index]
        file_counts[meta.label_text].add(str(meta.path))
        sample_counts[meta.label_text] += 1
    return {
        label: {
            "files": len(paths),
            "samples": sample_counts[label],
        }
        for label, paths in file_counts.items()
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = CWRUClipDataset(
        fault_root=args.fault_root,
        normal_root=args.normal_root,
        window_size=args.window_size,
        stride=args.stride,
        limit_per_file=args.limit_per_file,
    )
    class_texts = [dataset.idx_to_text[idx] for idx in range(len(dataset.idx_to_text))]
    class_tokens = tokenizer(
        class_texts,
        padding=True,
        truncation=True,
        max_length=args.max_text_len,
        return_tensors="pt",
    )
    train_set, val_set, test_set = build_grouped_splits(dataset, seed=args.seed)
    collate_fn = build_collate_fn(tokenizer, args.max_text_len)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_input_ids = class_tokens["input_ids"].to(device)
    class_attention_mask = class_tokens["attention_mask"].to(device)
    num_classes = len(dataset.idx_to_text)
    use_amp = device.type == "cuda" and not args.disable_amp
    model = FaultClipModel(
        signal_length=args.window_size,
        embed_dim=args.embed_dim,
        text_model_name=args.text_model,
        freeze_text_backbone=args.freeze_text_backbone,
        prompt_layers=args.prompt_layers,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0
    history = []
    split_summary = {
        "train": summarize_split(dataset, train_set),
        "val": summarize_split(dataset, val_set),
        "test": summarize_split(dataset, test_set),
    }
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as fp:
        json.dump(split_summary, fp, ensure_ascii=False, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        prompt_weight = args.prompt_loss_weight * min(1.0, epoch / max(args.prompt_loss_warmup_epochs, 1))

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(progress, start=1):
            signals = batch["signals"].to(device, non_blocking=True)
            label_indices = batch["label_indices"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                text_features, text_prompt_states = model.encode_text(
                    class_input_ids, class_attention_mask, return_prompt_states=True
                )
                signal_features, signal_prompt_states = model.encode_signal(signals, return_prompt_states=True)
                scale = model.logit_scale.exp().clamp(max=100.0)
                contrastive_loss, logits = class_contrastive_loss(signal_features, text_features, label_indices, scale)
                prompt_loss = prompt_alignment_loss(signal_prompt_states, text_prompt_states, label_indices)
                total_loss = contrastive_loss + prompt_weight * prompt_loss
                loss = total_loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            if step_idx % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * signals.size(0) * args.grad_accum_steps
            total += signals.size(0)
            progress.set_postfix(
                loss=f"{loss.item() * args.grad_accum_steps:.4f}",
                clip=f"{contrastive_loss.item():.4f}",
                prompt=f"{prompt_loss.item():.4f}",
                pw=f"{prompt_weight:.4f}",
                scale=f"{scale.item():.2f}",
            )

        if len(train_loader) % args.grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(total, 1)
        val_metrics = evaluate(model, val_loader, class_input_ids, class_attention_mask, device, num_classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_class_acc": val_metrics["accuracy"],
                "val_per_class_acc": val_metrics["class_accuracy"],
            }
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_class_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            checkpoint = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "label_texts": dataset.idx_to_text,
                "label_to_idx": dataset.label_to_idx,
                "split_summary": split_summary,
                "best_val_metrics": val_metrics,
            }
            torch.save(checkpoint, output_dir / "best.pt")

    if args.save_last:
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "label_texts": dataset.idx_to_text,
                "label_to_idx": dataset.label_to_idx,
                "split_summary": split_summary,
            },
            output_dir / "last.pt",
        )

    best_checkpoint_path = output_dir / "best.pt"
    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, class_input_ids, class_attention_mask, device, num_classes)
    print(f"Test: loss={test_metrics['loss']:.4f}, class_acc={test_metrics['accuracy']:.4f}")

    with open(output_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, ensure_ascii=False, indent=2)
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(test_metrics, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
