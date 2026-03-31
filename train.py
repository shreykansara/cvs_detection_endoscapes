"""
Two-Phase Training Script for CVS Detection
---------------------------------------------
Phase 1: Train only the classifier head (backbone frozen).
          Optimizer : AdamW, lr=1e-3
          Epochs    : 5

Phase 2: Fine-tune the entire network (backbone unfrozen).
          Optimizer : AdamW, lr=1e-4
          Scheduler : CosineAnnealingLR
          Epochs    : 25
          Best checkpoint saved by validation AUC.

Design decisions specific to this implementation:
  - AdamW (not SGD) — better convergence for pretrained vision models
  - pos_weight computed from actual split counts (not hardcoded)
  - Two-phase strategy to stabilise head before full fine-tuning
  - AUC as primary metric (clinically meaningful for imbalanced data)
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from dataset import make_loaders
from model import CVSClassifier


LABEL_MODE  = "binary"
BATCH_SIZE  = 32
NUM_WORKERS = 4
DROPOUT     = 0.4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42

LR_HEAD      = 1e-3
EPOCHS_P1    = 5
LR_FULL      = 1e-4
EPOCHS_P2    = 15
WEIGHT_DECAY = 1e-4


def evaluate_split(model, loader, criterion, device):
    """
    Runs inference on a dataloader and returns loss, AUC, and F1.
    All predictions are accumulated before computing metrics to
    ensure AUC is computed over the full split, not per-batch.
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.float().to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    auc   = roc_auc_score(all_labels, all_probs)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    f1    = f1_score(all_labels, preds, zero_division=0)
    return total_loss / len(loader), auc, f1


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch — returns loss, AUC, F1."""
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc="  Training", unit="batch", leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_probs.extend(
            torch.sigmoid(logits).detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    auc   = roc_auc_score(all_labels, all_probs)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    f1    = f1_score(all_labels, preds, zero_division=0)
    return total_loss / len(loader), auc, f1


def run_training():
    torch.manual_seed(SEED)
    print(f"\nDevice: {DEVICE}\n")

    train_loader, val_loader, test_loader, pos_weight = make_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        label_mode=LABEL_MODE)

    num_outputs = 3 if LABEL_MODE == "per_criterion" else 1
    model = CVSClassifier(
        dropout=DROPOUT,
        freeze_backbone=True,
        num_outputs=num_outputs).to(DEVICE)

    # Loss: BCEWithLogitsLoss with computed pos_weight
    # pos_weight upweights the minority CVS=1 class during training
    pw        = torch.tensor([pos_weight]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── Phase 1: head only ──────────────────────────────────────────
    print("\n=== Phase 1: Training classifier head (backbone frozen) ===")
    optimizer_p1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS_P1 + 1):
        tr_loss, tr_auc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer_p1, DEVICE)
        vl_loss, vl_auc, vl_f1 = evaluate_split(
            model, val_loader, criterion, DEVICE)
        print(f"  [P1 {epoch:02d}/{EPOCHS_P1}] "
              f"train → loss={tr_loss:.4f} AUC={tr_auc:.4f} F1={tr_f1:.4f}  |  "
              f"val   → loss={vl_loss:.4f} AUC={vl_auc:.4f} F1={vl_f1:.4f}")

    # ── Phase 2: full fine-tune ─────────────────────────────────────
    print("\n=== Phase 2: Full fine-tuning (backbone unfrozen) ===")
    model.unfreeze_backbone()
    optimizer_p2 = AdamW(
        model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer_p2, T_max=EPOCHS_P2)

    best_val_auc    = 0.0
    best_state      = None
    patience        = 5    # stop if no improvement for 5 consecutive epochs
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS_P2 + 1):
        tr_loss, tr_auc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer_p2, DEVICE)
        vl_loss, vl_auc, vl_f1 = evaluate_split(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"  [P2 {epoch:02d}/{EPOCHS_P2}] "
              f"train → loss={tr_loss:.4f} AUC={tr_auc:.4f} F1={tr_f1:.4f}  |  "
              f"val   → loss={vl_loss:.4f} AUC={vl_auc:.4f} F1={vl_f1:.4f}")

        if vl_auc > best_val_auc:
            best_val_auc      = vl_auc
            best_state        = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"    ✓ New best val AUC: {best_val_auc:.4f} — checkpoint saved")
        else:
            epochs_no_improve += 1
            print(f"    No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

    # ── Final evaluation on test set ───────────────────────────────
    print("\n=== Test set evaluation ===")
    model.load_state_dict(best_state)
    torch.save(best_state, "cvs_endoscapes_convnext_best.pth")

    ts_loss, ts_auc, ts_f1 = evaluate_split(
        model, test_loader, criterion, DEVICE)
    print(f"  Test AUC : {ts_auc:.4f}")
    print(f"  Test F1  : {ts_f1:.4f}")
    print(f"  Test loss: {ts_loss:.4f}")

    return model


if __name__ == "__main__":
    run_training()