import matplotlib.pyplot as plt
import numpy as np

def plot_fbeta_vs_threshold(
    thresholds,
    fbeta_vals,
    best_threshold,
    beta,
):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10,5))
    plt.plot(thresholds, fbeta_vals, label=f'F{beta}')
    plt.axvline(best_threshold, color='black', linestyle='--', linewidth=2, label=f'best thr={best_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel(f'F{beta} score')
    plt.title(f'F{beta} vs Threshold (from Optimization History)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_anomaly_distribution(normal_scores: np.ndarray, anomaly_scores: np.ndarray, threshold: float, title: str = "Anomaly Score Distribution"):
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
    plt.tight_layout()
    plt.show()

def plot_reconstructions(originals: np.ndarray, reconstructions: np.ndarray, title: str, n: int = 4):
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4))
    fig.suptitle(title, fontsize=14)

    for idx in range(n):
        axes[0, idx].imshow(originals[idx].reshape(28, 28), cmap='gray')
        axes[0, idx].axis('off')
        axes[1, idx].imshow(reconstructions[idx].reshape(28, 28), cmap='gray')
        axes[1, idx].axis('off')

    axes[0, 0].set_ylabel('Original', rotation=0, labelpad=35, va='center')
    axes[1, 0].set_ylabel('Recon', rotation=0, labelpad=35, va='center')
    plt.tight_layout()
    plt.show()
