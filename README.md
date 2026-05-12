## FD-CLSP

### Motivation

A simple CLIP-style alignment objective is effective, but during the signal-text alignment training process, there was an unstable training situation that did not converge, and the fault classification robustness was weak.
To achieve early cross-modal alignment without compromising the backbone features, we introduced a lightweight prompt injection mechanism during the training process.
To improve the robustness when inferring long vibration signals, we further aggregated multiple window features instead of relying on a single window.
### Training-Side Improvement: Stable Prompt Injection

The stable version injects a small prompt-guided residual into the **early layers** of both encoders:

- **Signal encoder**: uses a learnable domain prompt and current-layer signal features
- **Text encoder**: uses text-side prompt features and current-layer text representations
- The prompt branch is only applied to the **first encoder layer**
- Prompt alignment is optimized with an auxiliary loss in addition to the main contrastive loss

At each selected layer, a prompt is generated from:

- a static prompt representation
- the current hidden feature of that layer

The fused prompt is then injected into the next-layer input as a **small residual correction**.

### Why This Version Is Stable

Earlier prompt-injection attempts were too aggressive and could collapse class boundaries.  
The current stable version uses three constraints:

1. **Single-layer injection**: only the first layer is modified
2. **Tiny residual gate**: prompt influence starts very small
3. **Prompt-loss warmup**: the auxiliary prompt alignment loss is gradually introduced during early epochs

These changes let the backbone learn the main classification structure first, while the prompt branch improves alignment progressively.

### Training Objective

The final loss contains two parts:

- **Main loss**: class-level contrastive alignment loss
- **Auxiliary loss**: prompt alignment loss between signal-side and text-side prompt states

The prompt loss is introduced with a warmup schedule to avoid destabilizing early training.

### Inference-Side Improvement: KMeans Aggregation

For long vibration signals, a single window may be noisy or unrepresentative.  
To improve robustness, inference can be performed on multiple signal windows and then aggregated at the feature level.

Supported aggregation strategies include:

- `first`: use only the first window
- `mean`: average all window features
- `kmeans`: cluster all window features and use the largest cluster center for final matching

The `kmeans` strategy is designed to suppress outlier windows and emphasize the dominant fault pattern within a signal file.

### Practical Effect

Under the same grouped 4-class split:

- **Baseline model**: `93.21%` test accuracy
- **Stable prompt-injected model**: `99.62%` test accuracy

The improvement is especially clear for:

- `inner race fault`
- `outer race fault`

For file-level inference on long signals, multi-window aggregation further improves robustness compared with single-window prediction.

### Recommended Settings

Current stable training configuration:

- `prompt_layers = 1`
- `prompt_loss_weight = 0.01`
- `prompt_loss_warmup_epochs = 3`

Recommended robust inference setting:

- `aggregation = kmeans`
- `num_clusters = 3`

This combination gives a practical balance between training stability, cross-modal alignment quality, and inference robustness.
