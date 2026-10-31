"""CIFAR-10 / CIFAR-100 image classification, as in Hyper-SET (Phase C.2 Task 7).

Source:
    - paper: Hyper-SET (ICLR 2026), §5.2, Appendix D.3, Table 1, 9, 10
    - repo: hyper-set, ic/dataset.py (build_transforms, _DATASET_META), configs/task/ic.yaml,
      scripts/ic/train_ic.sh (CIFAR: model.patch=8)

Notes:
    - data: torchvision CIFAR10 / CIFAR100 (50,000 train, 10,000 test), downloaded to data/cifar/raw/
    - splits: "train" = the official train set, "test_id" = the official test set (no OOD split; Hyper-SET reports
      top-1 accuracy on the test set)
    - transforms as Hyper-SET (CIFAR, training from scratch, autoaugment true):
        train: RandomCrop(32, padding 4), RandomHorizontalFlip, AutoAugment (CIFAR-10 policy), ToTensor, Normalize
        test:  ToTensor, Normalize
      AutoAugment: torchvision's AutoAugment(CIFAR10) (the CIFAR-10 policy of Cubuk et al. 2019, the same 25
      sub-policies as Hyper-SET's ic/augmentation.py CIFAR10Policy); Hyper-SET uses it for CIFAR-100 too
    - one token per 8x8 patch: (3, 32, 32) -> 16 tokens of 8 * 8 * 3 = 192 values (channel-last inside a patch)
    - label smoothing 0.1 of Hyper-SET's loss is in the task model (models/tasks/cifar.py), not here
"""
import torch
import torchvision
import torchvision.transforms as T

from data.base import BaseDataset, register_dataset
from utils.logger import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "cifar" / "raw"
PATCH = 8


def patchify(img, patch=PATCH):
    """(C, H, W) -> (H/p * W/p, p * p * C), patches row by row, each patch flattened channel-last."""
    c, h, w = img.shape
    x = img.reshape(c, h // patch, patch, w // patch, patch)          # (C, h, p, w, p)
    return x.permute(1, 3, 2, 4, 0).reshape((h // patch) * (w // patch), patch * patch * c)


def unpatchify(tokens, patch=PATCH, size=32, channels=3):
    """Inverse of patchify: (N, p * p * C) -> (C, H, W)."""
    n = size // patch
    x = tokens.reshape(n, n, patch, patch, channels)                   # (h, w, p, p, C)
    return x.permute(4, 0, 2, 1, 3).reshape(channels, size, size)


class _CIFAR(BaseDataset):
    """Shared code of CIFAR10 / CIFAR100.

    - inputs: (16, 192) float, normalised patches; labels: () long, the class
    - metric: top-1 accuracy (%), key "acc"
    """

    SPLITS = ("train", "test_id")
    BEST_METRIC = ("loss", "min")
    PLOT_METRIC = ({"acc": "Top-1"}, "Accuracy", "Acc (%)", (0, 100))
    input_len = 16
    output_len = 1
    TV_CLASS = None
    MEAN = STD = None

    def __init__(self, split):
        super().__init__(split)
        train = split == "train"
        normalize = T.Normalize(self.MEAN, self.STD)
        if train:
            self.transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                                        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10), T.ToTensor(), normalize])
        else:
            self.transform = T.Compose([T.ToTensor(), normalize])
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.data = self.TV_CLASS(root=str(RAW_DIR), train=train, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        return {"inputs": patchify(self.transform(img)), "labels": torch.tensor(label, dtype=torch.long)}

    def metric(self, pred, labels):
        """pred: class indices (B,) or logits (B, C); labels (B,). Returns {"acc": top-1 accuracy in %}."""
        if pred.dim() == 2:
            pred = pred.argmax(dim=-1)
        return {"acc": 100.0 * (pred == labels).float().mean().item()}

    def sample_scores(self, pred, labels):
        if pred.dim() == 2:
            pred = pred.argmax(dim=-1)
        return {"correct": (pred == labels).float()}

    def metric_from_scores(self, scores):
        return {"acc": 100.0 * float(scores["correct"].sum() / scores["correct"].numel())}


@register_dataset("cifar10")
class CIFAR10(_CIFAR):
    TV_CLASS = torchvision.datasets.CIFAR10
    MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]                # Hyper-SET _DATASET_META c10


@register_dataset("cifar100")
class CIFAR100(_CIFAR):
    TV_CLASS = torchvision.datasets.CIFAR100
    MEAN, STD = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]                # Hyper-SET _DATASET_META c100
