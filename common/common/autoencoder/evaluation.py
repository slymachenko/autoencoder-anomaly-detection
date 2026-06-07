import torch
import torch.nn as nn
import numpy as np

from typing import Tuple
from torch.utils.data import DataLoader

def score_images(model, images, device, criterion = nn.functional.mse_loss):
    model.eval()
    images = images.reshape(images.shape[0], 1, 28, 28)

    if images.device != device:
        images = images.to(device)

    with torch.no_grad():
        reconstruction = model(images)
        pixel_errors = criterion(reconstruction, images, reduction="none")
        scores = torch.mean(pixel_errors, dim=(1, 2, 3)).cpu().numpy()

    return scores

def get_reconstructions(model, images, device, n=8):
    model.eval()
    sample_images = images.reshape(images.shape[0], 1, 28, 28)[:n].to(device)
    
    with torch.no_grad():
        reconstructions = model(sample_images).cpu()
    
    return sample_images.cpu().numpy(), reconstructions.numpy()

@torch.no_grad()
def evaluate_anomaly_ensemble(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Experiment 3: Test-Time Input Transformation Ensembling (Multi-rotation variance).
    Computes both standard reconstruction errors and variance across 4 canonical rotations.
    """
    model.eval()
    all_scores = []
    all_variances = []
    
    # Standard MSE fallback for individual pixel maps if no criterion is passed
    base_mse = nn.MSELoss(reduction='none')

    for data, _ in test_loader:
        img = data.to(device)
        batch_rotation_errors = []

        # Evaluate across 0, 90, 180, and 270 degrees
        for k in range(4):
            rotated = torch.rot90(img, k, dims=[-2, -1])
            recon_rotated = model(rotated)
            # Realign the output spatially back to original orientation
            recon_unrotated = torch.rot90(recon_rotated, -k, dims=[-2, -1])
            
            # Calculate sample spatial error maps (unreduced)
            pixel_error = base_mse(recon_unrotated, img) 
            batch_rotation_errors.append(pixel_error)
            
        # Shape: [4, Batch, Channels, H, W]
        stacked_errors = torch.stack(batch_rotation_errors)
        
        # 1. Anomaly Score: Mean reconstruction error across the ensemble
        mean_spatial_error = torch.mean(stacked_errors, dim=0)
        # Reduce to a single score per image in the batch (Mean across channels & spatial dimensions)
        image_anomaly_scores = mean_spatial_error.mean(dim=[-3, -2, -1])
        
        # 2. Test-Time Rotation Variance: High variance typically correlates with anomalies
        spatial_variance = torch.var(stacked_errors, dim=0)
        image_variance_scores = spatial_variance.mean(dim=[-3, -2, -1])

        all_scores.extend(image_anomaly_scores.cpu().numpy())
        all_variances.extend(image_variance_scores.cpu().numpy())

    return np.array(all_scores), np.array(all_variances)