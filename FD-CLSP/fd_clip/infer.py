from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.cluster import KMeans
from transformers import AutoTokenizer

from fd_clip.data import _normalize_signal, _window_signal
from fd_clip.model import FaultClipModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--signal-file", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--text-model", type=str, default=None)
    parser.add_argument("--aggregation", type=str, default="first", choices=["first", "mean", "kmeans"])
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def load_drive_end_signal(mat_path: str | Path) -> np.ndarray:
    mat = sio.loadmat(mat_path)
    key = next((name for name in mat if name.endswith("_DE_time")), None)
    if key is None:
        raise KeyError(f"No drive-end signal found in {mat_path}")
    signal = np.asarray(mat[key]).reshape(-1)
    return _normalize_signal(signal)


def build_signal_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows = list(_window_signal(signal, window_size=window_size, stride=stride))
    return np.stack(windows, axis=0)


@torch.no_grad()
def encode_signal_windows(model: FaultClipModel, windows: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    features = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
        features.append(model.encode_signal(batch).cpu())
    return torch.cat(features, dim=0)


def aggregate_signal_features(features: torch.Tensor, aggregation: str, num_clusters: int) -> torch.Tensor:
    if aggregation == "first":
        return features[0]
    if aggregation == "mean":
        return features.mean(dim=0)
    if aggregation == "kmeans":
        feature_array = features.numpy()
        k = max(1, min(num_clusters, len(feature_array)))
        if k == 1:
            center = feature_array.mean(axis=0)
        else:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            assignments = kmeans.fit_predict(feature_array)
            counts = np.bincount(assignments, minlength=k)
            largest_cluster = int(counts.argmax())
            center = kmeans.cluster_centers_[largest_cluster]
        center_tensor = torch.from_numpy(center).float()
        return center_tensor / center_tensor.norm(p=2).clamp(min=1e-12)
    raise ValueError(f"Unsupported aggregation: {aggregation}")


@torch.no_grad()
def predict_signal_file(
    model: FaultClipModel,
    tokenizer,
    label_texts: list[str],
    signal_file: str | Path,
    window_size: int,
    stride: int,
    aggregation: str,
    num_clusters: int,
    batch_size: int,
    device: torch.device,
) -> tuple[int, torch.Tensor]:
    signal = load_drive_end_signal(signal_file)
    windows = build_signal_windows(signal, window_size=window_size, stride=stride)
    signal_features = encode_signal_windows(model, windows, batch_size=batch_size, device=device)
    aggregated = aggregate_signal_features(signal_features, aggregation=aggregation, num_clusters=num_clusters).unsqueeze(0).to(device)

    tokens = tokenizer(label_texts, padding=True, truncation=True, max_length=24, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    text_features = model.encode_text(tokens["input_ids"], tokens["attention_mask"])

    scores = (aggregated @ text_features.t()).squeeze(0).cpu()
    probabilities = scores.softmax(dim=0)
    top_idx = int(probabilities.argmax().item())
    return top_idx, probabilities


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    text_model = args.text_model or train_args["text_model"]
    stride = args.stride or train_args.get("stride", args.window_size)

    tokenizer = AutoTokenizer.from_pretrained(text_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultClipModel(
        signal_length=args.window_size,
        embed_dim=train_args["embed_dim"],
        text_model_name=text_model,
        freeze_text_backbone=train_args["freeze_text_backbone"],
        prompt_layers=train_args.get("prompt_layers", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    label_texts = [text for _, text in sorted(checkpoint["label_texts"].items(), key=lambda item: int(item[0]))]
    top_idx, probabilities = predict_signal_file(
        model=model,
        tokenizer=tokenizer,
        label_texts=label_texts,
        signal_file=args.signal_file,
        window_size=args.window_size,
        stride=stride,
        aggregation=args.aggregation,
        num_clusters=args.num_clusters,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"Predicted label: {label_texts[top_idx]}")
    print(f"Aggregation: {args.aggregation}")
    print("Top probabilities:")
    for idx in probabilities.argsort(descending=True).tolist():
        print(f"  {label_texts[idx]}: {probabilities[idx].item():.4f}")


if __name__ == "__main__":
    main()
