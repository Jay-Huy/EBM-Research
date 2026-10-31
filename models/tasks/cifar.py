"""CIFAR-10 / CIFAR-100 (data/cifar/cifar_dataset.py), d_model = 512 (Hyper-SET Table 10: d 512, 8 heads).

- one token per 8x8 patch: inputs (B, 16, 192) -> X (B, 16, d) by a linear map; Z has 16 tokens too
- readout: mean of Z* over the 16 tokens -> nn.Linear(d, n_classes) -> logits (B, n_classes)
  (Hyper-SET reads a learnable [CLS] token; a mean keeps N = N_x for every task of ours)
- loss: cross-entropy with label smoothing 0.1 (Hyper-SET configs/task/ic.yaml)
"""
import math

import torch.nn as nn
import torch.nn.functional as F

from models.modeling_base import EnergyLoopModel, register_model

PATCH_DIM = 8 * 8 * 3
LABEL_SMOOTHING = 0.1


class MeanHead(nn.Module):
    """Z* (B, N, d) -> mean over the tokens -> logits (B, n_classes)."""

    def __init__(self, d_model, n_classes):
        super().__init__()
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, z):
        return self.proj(z.mean(dim=1))


class CIFARModel(EnergyLoopModel):
    """Encoder nn.Linear(192, d) no bias, decoder MeanHead, loss CE (label smoothing 0.1)."""

    N_CLASSES = None

    def build_encoder(self):
        return nn.Linear(PATCH_DIM, self.config.d_model, bias=False)

    def build_decoder(self):
        return MeanHead(self.config.d_model, self.N_CLASSES)

    def init_encoder_decoder(self, c=0.25):
        """- encoder: N(0, 1 / 192): a normalised patch (values ~1) gives a token with values ~1, the scale of Z0
        - decoder: trunc_normal std c / sqrt(d), +-3 std, as the core latent; bias 0
        """
        nn.init.normal_(self.encoder.weight, std=1.0 / math.sqrt(PATCH_DIM))
        std = c / math.sqrt(self.config.d_model)
        nn.init.trunc_normal_(self.decoder.proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.decoder.proj.bias)

    def compute_loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels, label_smoothing=LABEL_SMOOTHING)

    def predict(self, outputs):
        """Class index (B,), the format of the dataset metric."""
        return outputs.argmax(dim=-1)


class CIFAR10Model(CIFARModel):
    N_CLASSES = 10


class CIFAR100Model(CIFARModel):
    N_CLASSES = 100


register_model("cifar10", d_model=512, n_tokens=16, n_context=16)(CIFAR10Model)
register_model("cifar100", d_model=512, n_tokens=16, n_context=16)(CIFAR100Model)
