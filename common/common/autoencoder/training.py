import torch
import torch.nn as nn
from piq import ssim
import tqdm

from typing import List, Optional, Callable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor

def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    criterion_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    device: torch.device,
    loss_eps: float = 1e-4,
    patience: int = 5,
    max_epochs: int = 100,
    corruption_fn: Optional[Callable[[Tensor], Tensor]] = None,
    verbose: bool = False
) -> List[float]:
    """
    Args:
        criterion_fn: A callable: criterion_fn(reconstructed, target) -> scalar tensor.
        corruption_fn: An optional callable: corruption_fn(img_tensor) -> corrupted_img_tensor.
    """
    epoch_losses = []
    best_state = None
    best_epoch_loss = float("inf")

    pbar = tqdm.tqdm(desc="Training Autoencoder", disable=(not verbose), unit=" epoch")

    try:
        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0

            for data, _ in train_loader:
                clean_img = data.to(device) if data.device != device else data

                input_img = clean_img if corruption_fn is None else corruption_fn(clean_img)

                # Forward pass
                output = model(input_img)
                loss = criterion_fn(output, clean_img, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * clean_img.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_losses.append(epoch_loss)

            improvement = best_epoch_loss - epoch_loss
            if improvement > loss_eps:
                best_epoch_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            pbar.update(1)
            pbar.set_postfix({
                "Loss": f"{epoch_loss:.3e}",
                "Best": f"{best_epoch_loss:.3e}",
                "Patience": f"{patience_counter}/{patience}"
            })

            if patience_counter >= patience:
                pbar.set_description(f"[Early Stopped at Epoch {epoch+1}]")
                break
        else:
            pbar.set_description(f"[Hit Max Epochs]")
    finally:
        pbar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
        
    return epoch_losses

class HybridLoss(nn.Module):
    def __init__(self, alpha = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x_hat, x, reduction = "mean"):
        loss_mse = nn.functional.mse_loss(x_hat, x, reduction = reduction)
        if loss_mse.ndim > 1:
            loss_mse = torch.mean(loss_mse, dim=tuple([i for i in range(1, loss_mse.ndim)]))
        # SSIM is 1 for identical images, so we minimize (1 - (ssim + 1)/2) = (1 - ssim)/2
        loss_ssim = (1.0 - ssim(x_hat, x, data_range = 1.0, reduction = reduction)) / 2.0

        return (self.alpha * loss_mse) + ((1 - self.alpha) * loss_ssim)

class GaussianNoiseCorruption:
    def __init__(self, mean: float = 0.0, std: float = 0.2):
        self.mean = mean
        self.std = std

    def __call__(self, img_tensor):
        corrupted = img_tensor.clone()
        noise = torch.randn_like(corrupted) * self.std + self.mean
        corrupted = corrupted + noise
        return torch.clamp(corrupted, 0.0, 1.0)
