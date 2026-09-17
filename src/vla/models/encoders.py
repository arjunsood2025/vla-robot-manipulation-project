"""Vision and language encoders for the BC baseline.

Image encoder: ResNet-18 (ImageNet-initialised, fine-tuned). Small and fast so
    the baseline trains in minutes; the final FC is stripped and we use the 512-d
    global-pooled feature. Multiple camera views are encoded with the SAME shared
    encoder and concatenated — weight sharing across views is a sensible prior
    and keeps parameter count down on a small dataset.

Text encoder: the CLIP ViT-B/32 text tower, FROZEN. CLIP text features are a
    strong, general instruction embedding with no in-domain text to overfit to,
    and freezing them is the right call with only a few hundred demos — there is
    nowhere near enough language supervision to fine-tune a text model without
    destroying it. A learned text head over hundreds of instructions would just
    memorise them.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResNetImageEncoder(nn.Module):
    """Shared ResNet-18 trunk producing a 512-d feature per image."""

    OUT_DIM = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        # Drop the classifier; keep everything up to and including global avgpool.
        self.trunk = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, n_cam, 3, H, W) -> (B, n_cam * 512).

        Views are folded into the batch dimension so the shared trunk processes
        them in one pass, then unfolded and concatenated per sample.
        """
        b, n_cam = images.shape[:2]
        flat = images.flatten(0, 1)                 # (B*n_cam, 3, H, W)
        feat = self.trunk(flat).flatten(1)          # (B*n_cam, 512)
        return feat.view(b, n_cam * self.OUT_DIM)   # (B, n_cam*512)

    def output_dim(self, n_cameras: int) -> int:
        return n_cameras * self.OUT_DIM


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text tower. Tokenises raw strings and returns pooled features."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 cache_embeddings: bool = True):
        super().__init__()
        from transformers import CLIPTokenizer, CLIPTextModel

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_model.eval()
        self.out_dim = self.text_model.config.hidden_size  # 512 for ViT-B/32

        # The tower is frozen and always in eval mode, so a given string's
        # embedding is deterministic and worth caching: a control loop re-issues
        # the SAME instruction every tick, and without this each tick pays for
        # tokenisation plus a full text forward pass. Benchmarking the served BC
        # policy showed that dominating per-call latency. The cache is keyed by
        # string and an episode has one instruction, so it stays tiny.
        #
        # Caching stores each string encoded ALONE. Encoding the same string
        # inside a padded batch differs in the last ~1e-6 (float32 batched-matmul
        # nondeterminism, measurable with the cache disabled — it is not caused
        # by caching). Repeat calls at a fixed batch shape are bit-identical.
        # That is far below action-prediction error, but it does mean cached and
        # uncached runs are not bit-reproducible against each other; disable the
        # cache if you need exact parity with a previously batched run.
        self._cache_enabled = cache_embeddings
        self._cache: dict[str, torch.Tensor] = {}

    def _encode(self, instructions: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            instructions, padding=True, truncation=True, max_length=32, return_tensors="pt"
        ).to(device)
        out = self.text_model(**tokens)
        # Pooled output = features at the EOS token; CLIP's sentence embedding.
        return out.pooler_output  # (B, out_dim)

    @torch.no_grad()
    def forward(self, instructions: list[str], device: torch.device) -> torch.Tensor:
        if not self._cache_enabled:
            return self._encode(instructions, device)

        missing = [s for s in dict.fromkeys(instructions) if s not in self._cache]
        for s in missing:
            # Encode each new string on its own, so the value stored for a short
            # instruction can never depend on how long its batch-mates were.
            self._cache[s] = self._encode([s], device)[0]
        cached = [self._cache[s] for s in instructions]
        return torch.stack(cached).to(device)

    def clear_cache(self) -> None:
        self._cache.clear()

    def output_dim(self) -> int:
        return self.out_dim
