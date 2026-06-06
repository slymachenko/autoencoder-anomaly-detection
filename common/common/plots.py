from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional
from pathlib import Path

def plot_fbeta_vs_threshold(
    thresholds: np.ndarray,
    fbeta_vals: np.ndarray,
    best_threshold: float,
    beta: float,
    save_path: Optional[str | Path] = None
):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, fbeta_vals, label=f'F{beta}')
    ax.axvline(best_threshold, color='black', linestyle='--', linewidth=2, label=f'best thr={best_threshold:.4f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel(f'F{beta} score')
    ax.set_title(f'F{beta} vs Threshold (from Optimization History)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_anomaly_distribution(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        threshold: float,
        title: str = "Anomaly Score Distribution",
        save_path: Optional[str | Path] = None
):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(
        min(normal_scores.min().item(), anomaly_scores.min().item()),
        max(normal_scores.max().item(), anomaly_scores.max().item()),
        80,
    )
    ax.hist(normal_scores, bins=bins, alpha=0.7, label='normal')
    ax.hist(anomaly_scores, bins=bins, alpha=0.7, label='anomaly')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='threshold')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_reconstructions(
        originals: np.ndarray,
        reconstructions: np.ndarray,
        title: str,
        n: int = 4,
        save_path: Optional[str | Path] = None
):
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4))
    fig.suptitle(title, fontsize=14)

    for idx in range(n):
        axes[0, idx].imshow(originals[idx].reshape(28, 28), cmap='gray')
        axes[0, idx].axis('off')
        axes[1, idx].imshow(reconstructions[idx].reshape(28, 28), cmap='gray')
        axes[1, idx].axis('off')

    axes[0, 0].set_ylabel('Original', rotation=0, labelpad=35, va='center')
    axes[1, 0].set_ylabel('Recon', rotation=0, labelpad=35, va='center')
    fig.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_performance_curves(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str | Path] = None
):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve plot
    ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax[0].plot([0, 1], [0, 1], lw=2, linestyle='--')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Receiver Operating Characteristic (ROC)')
    ax[0].legend(loc="lower right")
    
    # PR Curve plot
    ax[1].plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall (PR) Curve')
    ax[1].legend(loc="lower left")
    
    fig.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")