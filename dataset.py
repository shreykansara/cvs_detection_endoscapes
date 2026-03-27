"""
Endoscapes CVS Dataset Loader
------------------------------
Loads frames and CVS labels from the Endoscapes2023 dataset
(Murali et al., arXiv:2312.12429) using all_metadata.csv.

Label source: C1, C2, C3 columns — per-criterion averages across
3 expert annotators, as described in the Endoscapes technical report.
Only frames where is_ds_keyframe == True carry CVS annotations.

This module is written from scratch for the Endoscapes dataset structure.
It is not derived from or based on any existing CVS codebase.
"""

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

ROOT = Path("C:/endoscapes")


def load_split(split: str, label_mode: str = "binary") -> list:
    """
    Reads all_metadata.csv and returns (img_path, label) pairs
    for keyframes belonging to the given split.

    The split membership is determined by {split}_vids.txt which
    lists the video IDs for each partition. Video IDs are stored
    in scientific notation floats and are cast via int(float(...)).

    label_mode options:
        'binary'        — 1 if mean(C1, C2, C3) >= 0.5, else 0
        'soft'          — float mean of C1, C2, C3 in [0, 1]
        'per_criterion' — list [C1, C2, C3] as floats
    """
    vids_file = ROOT / f"{split}_vids.txt"
    with open(vids_file) as f:
        split_vids = set(int(float(v.strip())) for v in f if v.strip())

    df = pd.read_csv(ROOT / "all_metadata.csv")

    # Restrict to labeled keyframes in this split
    df = df[
        (df["is_ds_keyframe"] == True) &
        (df["vid"].isin(split_vids))
    ].copy()

    # Compute labels from the averaged per-criterion scores
    if label_mode == "binary":
        mean_cvs = (df["C1"] + df["C2"] + df["C3"]) / 3.0
        df["label"] = (mean_cvs >= 0.5).astype(int)
    elif label_mode == "soft":
        df["label"] = (df["C1"] + df["C2"] + df["C3"]) / 3.0
    elif label_mode == "per_criterion":
        pass  # labels extracted row-by-row below
    else:
        raise ValueError(f"Unsupported label_mode: '{label_mode}'")

    samples = []
    missing = 0
    for _, row in df.iterrows():
        img_path = ROOT / split / f"{int(row['vid'])}_{int(row['frame'])}.jpg"
        if not img_path.exists():
            missing += 1
            continue

        if label_mode == "per_criterion":
            label = [float(row["C1"]), float(row["C2"]), float(row["C3"])]
        else:
            label = row["label"]

        samples.append((img_path, label))

    print(f"  [{split}] {len(samples)} frames loaded "
          f"({missing} missing on disk)")
    return samples


class EndoscapesCVSDataset(Dataset):
    """
    PyTorch Dataset for single-frame CVS classification on Endoscapes2023.

    Augmentation strategy for training frames:
      - Upscale to 320x320 then RandomCrop to 300x300 for scale variation
      - Random horizontal flip (valid: laparoscope orientation can vary)
      - ColorJitter to handle lighting variation in OR environments
      - Random rotation ±10° for camera angle variation
      - RandomAutocontrast to simulate specular highlight variation
      - ImageNet normalization (standard for pretrained EfficientNet)

    Validation/test frames are only resized to 300x300 — no augmentation.
    """

    def __init__(self, split: str, label_mode: str = "binary"):
        self.label_mode = label_mode
        self.samples = load_split(split, label_mode)

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.RandomCrop(300),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomRotation(10),
                transforms.RandomAutocontrast(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))

        if self.label_mode == "binary":
            return img, torch.tensor(label, dtype=torch.long)
        elif self.label_mode == "soft":
            return img, torch.tensor(label, dtype=torch.float32)
        elif self.label_mode == "per_criterion":
            return img, torch.tensor(label, dtype=torch.float32)


def make_loaders(batch_size: int = 32,
                 num_workers: int = 4,
                 label_mode: str = "binary"):
    """
    Builds train, val, and test DataLoaders.

    For binary mode, training uses WeightedRandomSampler to oversample
    the minority CVS=1 class (~11% of train set), so the model sees a
    roughly balanced stream of batches during training.

    pos_weight for BCEWithLogitsLoss is computed and returned here so
    train.py can pass it directly to the loss function — it is derived
    from the actual split counts, not hardcoded.
    """
    print("\nLoading splits...")
    train_ds = EndoscapesCVSDataset("train", label_mode)
    val_ds   = EndoscapesCVSDataset("val",   label_mode)
    test_ds  = EndoscapesCVSDataset("test",  label_mode)

    if label_mode == "binary":
        # Compute pos_weight = n_negative / n_positive from training set
        train_labels = [s[1] for s in train_ds.samples]
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        pos_weight = n_neg / n_pos
        print(f"\n  Class balance (train): "
              f"CVS=1: {n_pos} ({n_pos/len(train_labels):.1%})  "
              f"CVS=0: {n_neg}")
        print(f"  Computed pos_weight for loss: {pos_weight:.2f}")

        # WeightedRandomSampler: each sample's weight = 1 / class_count
        counts  = torch.bincount(torch.tensor(train_labels)).float()
        weights = 1.0 / counts[train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True)
    else:
        pos_weight = None
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, pos_weight


if __name__ == "__main__":
    print("Verifying Endoscapes dataset loaders...\n")
    train_loader, val_loader, test_loader, pw = make_loaders(batch_size=8)
    imgs, labels = next(iter(train_loader))
    print(f"\nSample batch shape : {imgs.shape}")
    print(f"Sample labels      : {labels.tolist()}")
    print(f"pos_weight         : {pw:.2f}")