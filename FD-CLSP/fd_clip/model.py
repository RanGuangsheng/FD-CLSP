from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PromptInjection(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.normal_(self.fusion.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fusion.bias)
        self.gate = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

    def forward(self, static_prompt: torch.Tensor, pooled_hidden: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([static_prompt, pooled_hidden], dim=-1)
        prompt = torch.tanh(self.fusion(fused))
        return self.gate * prompt


class SignalTransformerEncoder(nn.Module):
    def __init__(
        self,
        signal_length: int = 2048,
        patch_size: int = 16,
        emb_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        out_dim: int = 256,
        prompt_layers: int = 2,
    ) -> None:
        super().__init__()
        if signal_length % patch_size != 0:
            raise ValueError("signal_length must be divisible by patch_size")

        self.patch_size = patch_size
        self.num_layers = num_layers
        self.prompt_layers = min(prompt_layers, num_layers)
        self.patch_embed = nn.Linear(patch_size, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_encoding = PositionalEncoding(emb_dim, max_len=(signal_length // patch_size) + 1)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=emb_dim,
                    nhead=num_heads,
                    dim_feedforward=emb_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.proj = nn.Linear(emb_dim, out_dim)
        self.prompt_proj = nn.Linear(emb_dim, out_dim)
        self.static_prompt = nn.Parameter(torch.zeros(1, emb_dim))
        self.prompt_generators = nn.ModuleList([PromptInjection(emb_dim) for _ in range(self.prompt_layers)])

    def forward(self, signal: torch.Tensor, return_prompt_states: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size = signal.size(0)
        patches = signal.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        tokens = self.patch_embed(patches)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat([cls_tokens, tokens], dim=1)
        hidden_states = self.pos_encoding(hidden_states)

        prompt_states: list[torch.Tensor] = []
        static_prompt = self.static_prompt.expand(batch_size, -1)
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < self.prompt_layers:
                pooled_hidden = hidden_states[:, 0]
                prompt = self.prompt_generators[layer_idx](static_prompt, pooled_hidden)
                hidden_states = hidden_states + prompt.unsqueeze(1)
                prompt_states.append(prompt)
            hidden_states = layer(hidden_states)

        pooled = self.norm(hidden_states[:, 0])
        projected = self.proj(pooled)
        if return_prompt_states:
            return projected, prompt_states
        return projected


class TextGPT2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        out_dim: int = 256,
        freeze_backbone: bool = False,
        prompt_layers: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.prompt_layers = min(prompt_layers, len(self.backbone.h))
        hidden_dim = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.prompt_proj = nn.Linear(hidden_dim, out_dim)
        self.prompt_generators = nn.ModuleList([PromptInjection(hidden_dim) for _ in range(self.prompt_layers)])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_prompt_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)
        hidden_states = self.backbone.wte(input_ids) + self.backbone.wpe(position_ids)
        hidden_states = self.backbone.drop(hidden_states)

        attention_bias = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        attention_bias = (1.0 - attention_bias) * torch.finfo(hidden_states.dtype).min
        static_prompt = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

        prompt_states: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.backbone.h):
            if layer_idx < self.prompt_layers:
                pooled_hidden = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                    dim=1, keepdim=True
                ).clamp(min=1)
                prompt = self.prompt_generators[layer_idx](static_prompt, pooled_hidden)
                hidden_states = hidden_states + prompt.unsqueeze(1)
                prompt_states.append(prompt)

            block_outputs = block(
                hidden_states,
                attention_mask=attention_bias,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = block_outputs[0] if isinstance(block_outputs, tuple) else block_outputs

        hidden_states = self.backbone.ln_f(hidden_states)
        pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        projected = self.proj(pooled)
        if return_prompt_states:
            return projected, prompt_states
        return projected


class FaultClipModel(nn.Module):
    def __init__(
        self,
        signal_length: int = 2048,
        embed_dim: int = 256,
        signal_layers: int = 8,
        text_model_name: str = "gpt2",
        freeze_text_backbone: bool = False,
        prompt_layers: int = 2,
    ) -> None:
        super().__init__()
        self.signal_encoder = SignalTransformerEncoder(
            signal_length=signal_length,
            out_dim=embed_dim,
            num_layers=signal_layers,
            prompt_layers=prompt_layers,
        )
        self.text_encoder = TextGPT2Encoder(
            model_name=text_model_name,
            out_dim=embed_dim,
            freeze_backbone=freeze_text_backbone,
            prompt_layers=prompt_layers,
        )
        self.prompt_layers = prompt_layers
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

    def encode_signal(
        self, signal: torch.Tensor, return_prompt_states: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        outputs = self.signal_encoder(signal, return_prompt_states=return_prompt_states)
        if return_prompt_states:
            features, prompt_states = outputs
            projected_prompts = [self.signal_encoder.prompt_proj(prompt) for prompt in prompt_states]
            return F.normalize(features, dim=-1), projected_prompts
        return F.normalize(outputs, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_prompt_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        outputs = self.text_encoder(input_ids, attention_mask, return_prompt_states=return_prompt_states)
        if return_prompt_states:
            features, prompt_states = outputs
            projected_prompts = [self.text_encoder.prompt_proj(prompt) for prompt in prompt_states]
            return F.normalize(features, dim=-1), projected_prompts
        return F.normalize(outputs, dim=-1)

    def forward(
        self,
        signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal_features = self.encode_signal(signal)
        text_features = self.encode_text(input_ids, attention_mask)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_signal = scale * signal_features @ text_features.t()
        logits_per_text = logits_per_signal.t()
        return logits_per_signal, logits_per_text, scale


def clip_loss(logits_per_signal: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(logits_per_signal.size(0), device=logits_per_signal.device)
    loss_signal = F.cross_entropy(logits_per_signal, targets)
    loss_text = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_signal + loss_text)


def class_contrastive_loss(
    signal_features: torch.Tensor,
    text_features: torch.Tensor,
    label_indices: torch.Tensor,
    temperature_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = temperature_scale * signal_features @ text_features.t()
    loss = F.cross_entropy(logits, label_indices)
    return loss, logits


def prompt_alignment_loss(
    signal_prompt_states: list[torch.Tensor],
    text_prompt_states: list[torch.Tensor],
    label_indices: torch.Tensor,
) -> torch.Tensor:
    if not signal_prompt_states or not text_prompt_states:
        device = label_indices.device
        return torch.zeros((), device=device)

    num_layers = min(len(signal_prompt_states), len(text_prompt_states))
    losses = []
    for layer_idx in range(num_layers):
        target_prompts = text_prompt_states[layer_idx][label_indices]
        cosine = F.cosine_similarity(signal_prompt_states[layer_idx], target_prompts, dim=-1, eps=1e-8)
        losses.append(1.0 - cosine.mean())
    return torch.stack(losses).mean()
