from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from fd_clip.data import CWRUClipDataset, build_grouped_kfold_splits
from fd_clip.model import FaultClipModel, class_contrastive_loss, prompt_alignment_loss
from fd_clip.train import build_collate_fn, evaluate, set_seed, summarize_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault-root", type=str, required=True)
    parser.add_argument("--normal-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--text-model", type=str, default="pretrained\\gpt2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-text-len", type=int, default=24)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--freeze-text-backbone", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--prompt-layers", type=int, default=1)
    parser.add_argument("--prompt-loss-weight", type=float, default=0.01)
    parser.add_argument("--prompt-loss-warmup-epochs", type=int, default=3)
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_fold(
    dataset: CWRUClipDataset,
    train_set,
    val_set,
    test_set,
    tokenizer,
    args: argparse.Namespace,
    fold_seed: int,
) -> dict[str, object]:
    set_seed(fold_seed)
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

    class_texts = [dataset.idx_to_text[idx] for idx in range(len(dataset.idx_to_text))]
    class_tokens = tokenizer(
        class_texts,
        padding=True,
        truncation=True,
        max_length=args.max_text_len,
        return_tensors="pt",
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
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        prompt_weight = args.prompt_loss_weight * min(1.0, epoch / max(args.prompt_loss_warmup_epochs, 1))
        progress = tqdm(train_loader, desc=f"Fold train epoch {epoch}/{args.epochs}", leave=False)
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
                contrastive_loss, _ = class_contrastive_loss(signal_features, text_features, label_indices, scale)
                prompt_loss = prompt_alignment_loss(signal_prompt_states, text_prompt_states, label_indices)
                loss = (contrastive_loss + prompt_weight * prompt_loss) / args.grad_accum_steps

            scaler.scale(loss).backward()
            if step_idx % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * signals.size(0) * args.grad_accum_steps
            total += signals.size(0)

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
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best state captured during fold training.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, class_input_ids, class_attention_mask, device, num_classes)

    return {
        "history": history,
        "test_metrics": test_metrics,
        "split_summary": {
            "train": summarize_split(dataset, train_set),
            "val": summarize_split(dataset, val_set),
            "test": summarize_split(dataset, test_set),
        },
    }


def main() -> None:
    args = parse_args()
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
    )
    folds = build_grouped_kfold_splits(dataset, num_folds=args.num_folds, seed=args.seed)

    fold_results = []
    accuracies = []
    losses = []
    for fold_idx, (train_set, val_set, test_set) in enumerate(folds, start=1):
        print(f"=== Fold {fold_idx}/{args.num_folds} ===")
        fold_result = train_one_fold(
            dataset=dataset,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            tokenizer=tokenizer,
            args=args,
            fold_seed=args.seed + fold_idx,
        )
        fold_result["fold"] = fold_idx
        accuracies.append(fold_result["test_metrics"]["accuracy"])
        losses.append(fold_result["test_metrics"]["loss"])
        fold_results.append(fold_result)

    summary = {
        "num_folds": args.num_folds,
        "window_level_test_accuracy_mean": float(np.mean(accuracies)),
        "window_level_test_accuracy_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "window_level_test_accuracy_var": float(np.var(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "window_level_test_loss_mean": float(np.mean(losses)),
        "window_level_test_loss_std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
        "fold_test_accuracies": accuracies,
        "fold_test_losses": losses,
    }

    with open(output_dir / "crossval_window_results.json", "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "folds": fold_results}, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
