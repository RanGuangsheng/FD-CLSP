from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from fd_clip.data import CWRUClipDataset, build_grouped_splits
from fd_clip.infer import predict_signal_file
from fd_clip.model import FaultClipModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text-model", type=str, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=["first", "mean", "kmeans"],
        choices=["first", "mean", "kmeans"],
    )
    return parser.parse_args()


def evaluate_method(
    model: FaultClipModel,
    tokenizer,
    label_texts: list[str],
    dataset: CWRUClipDataset,
    subset,
    window_size: int,
    stride: int,
    aggregation: str,
    num_clusters: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    file_targets: dict[str, int] = {}
    for index in subset.indices:
        meta = dataset.sample_metas[index]
        file_targets[str(meta.path)] = meta.label_idx

    confusion = torch.zeros((len(label_texts), len(label_texts)), dtype=torch.long)
    per_file_predictions = {}

    for file_path, target_idx in sorted(file_targets.items()):
        pred_idx, probabilities = predict_signal_file(
            model=model,
            tokenizer=tokenizer,
            label_texts=label_texts,
            signal_file=file_path,
            window_size=window_size,
            stride=stride,
            aggregation=aggregation,
            num_clusters=num_clusters,
            batch_size=batch_size,
            device=device,
        )
        confusion[target_idx, pred_idx] += 1
        per_file_predictions[file_path] = {
            "target": label_texts[target_idx],
            "predicted": label_texts[pred_idx],
            "probabilities": {
                label_texts[idx]: float(probabilities[idx].item()) for idx in range(len(label_texts))
            },
        }

    correct = int(confusion.diag().sum().item())
    total = int(confusion.sum().item())
    class_accuracy = {}
    for class_idx, label_text in enumerate(label_texts):
        class_total = int(confusion[class_idx].sum().item())
        class_accuracy[label_text] = 0.0 if class_total == 0 else float(confusion[class_idx, class_idx].item() / class_total)

    return {
        "aggregation": aggregation,
        "accuracy": 0.0 if total == 0 else correct / total,
        "confusion_matrix": confusion.tolist(),
        "class_accuracy": class_accuracy,
        "file_predictions": per_file_predictions,
    }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    text_model = args.text_model or train_args["text_model"]
    window_size = args.window_size or train_args["window_size"]
    stride = args.stride or train_args["stride"]

    dataset = CWRUClipDataset(
        fault_root=train_args["fault_root"],
        normal_root=train_args.get("normal_root"),
        window_size=window_size,
        stride=stride,
        limit_per_file=train_args.get("limit_per_file"),
    )
    _, _, test_set = build_grouped_splits(dataset, seed=train_args["seed"])
    label_texts = [text for _, text in sorted(checkpoint["label_texts"].items(), key=lambda item: int(item[0]))]

    tokenizer = AutoTokenizer.from_pretrained(text_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultClipModel(
        signal_length=window_size,
        embed_dim=train_args["embed_dim"],
        text_model_name=text_model,
        freeze_text_backbone=train_args["freeze_text_backbone"],
        prompt_layers=train_args.get("prompt_layers", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    results = {"test_files": len({str(dataset.sample_metas[i].path) for i in test_set.indices}), "methods": {}}
    for aggregation in args.aggregations:
        results["methods"][aggregation] = evaluate_method(
            model=model,
            tokenizer=tokenizer,
            label_texts=label_texts,
            dataset=dataset,
            subset=test_set,
            window_size=window_size,
            stride=stride,
            aggregation=aggregation,
            num_clusters=args.num_clusters,
            batch_size=args.batch_size,
            device=device,
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
