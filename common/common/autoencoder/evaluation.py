import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import itertools
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    fbeta_score, 
    confusion_matrix, 
    classification_report
)

def score_images(
        model,
        images,
        device,
        criterion_fn = nn.functional.mse_loss,
        reduction = "none"
):
    model.eval()
    images = images.reshape(images.shape[0], 1, 28, 28)

    if images.device != device:
        images = images.to(device)

    with torch.no_grad():
        reconstruction = model(images)
        scores = criterion_fn(reconstruction, images, reduction = reduction)
        if scores.ndim > 1:
            scores = torch.mean(scores, dim=tuple([i for i in range(1, scores.ndim)]))

    return scores.cpu().numpy()

def get_reconstructions(model, images, device, n=8):
    model.eval()
    sample_images = images.reshape(images.shape[0], 1, 28, 28)[:n].to(device)
    
    with torch.no_grad():
        reconstructions = model(sample_images).cpu()
    
    return sample_images.cpu().numpy(), reconstructions.numpy()

def print_metrics(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    threshold: float, 
    beta: float = 2.0
):
    """Prints a comprehensive evaluation report."""
    y_pred = (y_scores > threshold).astype(int)

    print(f"--- Evaluation Metrics ---")
    print(f"Best threshold for F{beta}-score: {threshold:.6f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F{beta} score:  {fbeta_score(y_true, y_pred, beta=beta):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

def score_images_variation(model, images, device, criterion_fn, reduction="none"):
    """
    Evaluates anomaly confidence using Test-Time Augmentation (TTA).
    Calculates the variance of errors of all unique pairwise combinations of reconstructed images.
    """
    model.eval()
    images = images.reshape(images.shape[0], 1, 28, 28)

    if images.device != device:
        images = images.to(device)

    angles = [-40, -30, -20, 0, 20, 30, 40]
    reconstructions = []

    # Collect all reconstructions across all angles
    with torch.no_grad():
        for angle in angles:
            rotated_imgs = TF.rotate(images, angle, interpolation=InterpolationMode.BILINEAR)
            recon = model(rotated_imgs)
            reconstructions.append(recon)

    # Compute all unique pairwise errors between reconstructions
    pairwise_errors = []
    for recon_a, recon_b in itertools.combinations(reconstructions, 2):
        pair_error = criterion_fn(recon_a, recon_b, reduction=reduction)
        if pair_error.ndim > 1:
            pair_error = torch.mean(pair_error, dim=tuple([i for i in range(1, pair_error.ndim)]))
        pairwise_errors.append(pair_error)

    stacked_pairwise = torch.stack(pairwise_errors, dim=0)
    mean_pairwise_scores = torch.var(stacked_pairwise, dim=0)

    return mean_pairwise_scores.cpu().numpy()