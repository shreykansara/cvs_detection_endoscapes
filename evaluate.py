"""
CVS Detection — Test Set Evaluation + Threshold Tuning + Grad-CAM
------------------------------------------------------------------
Loads the best saved checkpoint and evaluates on the Endoscapes test set.

Three outputs:
  1. Full classification report at optimal threshold
  2. Threshold tuning curve (sensitivity vs specificity)
  3. Grad-CAM heatmap overlays on sample frames
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)

from dataset import make_loaders, EndoscapesCVSDataset, ROOT
from model import CVSClassifier


DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT  = "cvs_endoscapes_best.pth"
LABEL_MODE  = "binary"
BATCH_SIZE  = 32
NUM_WORKERS = 4


# ── 1. Load model ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> CVSClassifier:
    model = CVSClassifier(dropout=0.0, freeze_backbone=False, num_outputs=1)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


# ── 2. Collect all predictions ───────────────────────────────────────────────

def get_predictions(model, loader):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            probs = torch.sigmoid(model(imgs)).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())
    return np.array(all_probs), np.array(all_labels)


# ── 3. Find optimal threshold ────────────────────────────────────────────────

def find_best_threshold(probs, labels):
    """
    Sweeps thresholds from 0.1 to 0.9 and picks the one that
    maximises F1 on the validation set. For surgical safety,
    you may prefer to maximise sensitivity instead — see comments.
    """
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t
    return round(float(best_thresh), 2), round(best_f1, 4)


# ── 4. Full evaluation report ────────────────────────────────────────────────

def evaluate(model, test_loader, val_loader):
    print("\n=== Collecting predictions ===")
    test_probs,  test_labels  = get_predictions(model, test_loader)
    val_probs,   val_labels   = get_predictions(model, val_loader)

    # Find optimal threshold on validation set, apply to test set
    best_thresh, _ = find_best_threshold(val_probs, val_labels)
    print(f"Optimal threshold (from val set): {best_thresh}")

    test_preds = (test_probs >= best_thresh).astype(int)

    print("\n=== Test Set Results ===")
    print(f"AUC  : {roc_auc_score(test_labels, test_probs):.4f}")
    print(f"F1   : {f1_score(test_labels, test_preds, zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=["No CVS", "CVS achieved"],
        zero_division=0))

    cm = confusion_matrix(test_labels, test_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Sensitivity (recall for CVS=1) : {sensitivity:.4f}")
    print(f"Specificity (recall for CVS=0) : {specificity:.4f}")
    print(f"\nConfusion matrix:\n{cm}")

    plot_roc_and_pr(test_labels, test_probs, best_thresh)
    return best_thresh


# ── 5. ROC + Precision-Recall curves ────────────────────────────────────────

def plot_roc_and_pr(labels, probs, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    axes[0].plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — CVS Detection")
    axes[0].legend()

    # Precision-Recall curve
    prec, rec, thresholds = precision_recall_curve(labels, probs)
    axes[1].plot(rec, prec, color="darkorange", lw=2)
    # Mark the chosen threshold
    idx = np.argmin(np.abs(thresholds - threshold))
    axes[1].scatter(rec[idx], prec[idx], color="red", zorder=5,
                    label=f"Threshold = {threshold}")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — CVS Detection")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cvs_roc_pr_curves.png", dpi=150)
    plt.show()
    print("Saved: cvs_roc_pr_curves.png")


# ── 6. Grad-CAM ──────────────────────────────────────────────────────────────

def compute_gradcam(model, img_tensor):
    """
    Generates a Grad-CAM heatmap for a single image tensor.
    Hooks onto the last convolutional block of EfficientNet-B3.
    Returns a heatmap array normalised to [0, 1].
    """
    model.eval()
    activations, gradients = [], []

    target_layer = model.backbone.features[-1]
    fwd_hook = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o.detach()))
    bwd_hook = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach()))

    img   = img_tensor.unsqueeze(0).to(DEVICE)
    logit = model(img)
    model.zero_grad()
    logit.backward()

    fwd_hook.remove()
    bwd_hook.remove()

    act  = activations[0].squeeze()        # (C, H, W)
    grad = gradients[0].squeeze()          # (C, H, W)
    weights = grad.mean(dim=(1, 2))        # global average pool gradients
    cam  = (weights[:, None, None] * act).sum(0)
    cam  = torch.relu(cam).cpu().numpy()
    cam  = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam


def visualise_gradcam(model, threshold, n_samples=6):
    """
    Picks n_samples frames from the test set (mix of TP, FP, FN, TN)
    and saves Grad-CAM overlays to disk.
    """
    test_ds = EndoscapesCVSDataset("test", label_mode=LABEL_MODE)

    # Inference transform (no augmentation)
    infer_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Collect samples: try to get a mix of outcomes
    results = {"TP": [], "TN": [], "FP": [], "FN": []}
    for img_path, true_label in test_ds.samples:
        if sum(len(v) for v in results.values()) >= n_samples * 4:
            break
        img_tensor = infer_transform(
            Image.open(img_path).convert("RGB"))
        with torch.no_grad():
            prob = torch.sigmoid(
                model(img_tensor.unsqueeze(0).to(DEVICE))).item()
        pred = int(prob >= threshold)

        outcome = ("TP" if true_label == 1 and pred == 1 else
                   "TN" if true_label == 0 and pred == 0 else
                   "FP" if true_label == 0 and pred == 1 else "FN")
        if len(results[outcome]) < n_samples:
            results[outcome].append((img_path, img_tensor, true_label, prob, outcome))

    # Plot
    all_samples = [s for group in results.values() for s in group][:n_samples * 2]
    fig, axes = plt.subplots(
        len(all_samples), 2,
        figsize=(8, 4 * len(all_samples)))

    for i, (img_path, img_tensor, true_label, prob, outcome) in enumerate(all_samples):
        # Original image
        orig = np.array(Image.open(img_path).convert("RGB").resize((300, 300)))

        # Grad-CAM heatmap
        cam = compute_gradcam(model, img_tensor)
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (300, 300), Image.BILINEAR)) / 255.0
        heatmap = cm.jet(cam_resized)[..., :3]
        overlay = (0.5 * orig / 255.0 + 0.5 * heatmap)
        overlay = np.clip(overlay, 0, 1)

        axes[i][0].imshow(orig)
        axes[i][0].set_title(
            f"{outcome} | true={true_label} | prob={prob:.2f}",
            fontsize=9)
        axes[i][0].axis("off")

        axes[i][1].imshow(overlay)
        axes[i][1].set_title("Grad-CAM", fontsize=9)
        axes[i][1].axis("off")

    plt.suptitle("CVS Detection — Grad-CAM Overlays", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("cvs_gradcam.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: cvs_gradcam.png")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _, val_loader, test_loader, _ = make_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        label_mode=LABEL_MODE)

    model = load_model(CHECKPOINT)

    # Full metrics report + ROC/PR curves
    best_threshold = evaluate(model, test_loader, val_loader)

    # Grad-CAM overlays on test frames
    visualise_gradcam(model, threshold=best_threshold, n_samples=6)