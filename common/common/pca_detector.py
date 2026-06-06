from sklearn.decomposition import PCA
import torch

from typing import Any

class PCAAnomalyDetector(PCA):
    def __init__(
        self,
        n_components: int | float | str | None = None,
        random_state: int | None = None
    ):
        super().__init__(n_components=n_components, random_state=random_state)

    def fit(self, X: torch.Tensor, y: Any = None):
        X_flat = X.reshape(X.shape[0], -1)
        super().fit(X_flat, y)
        return self

    def reconstruct(self, X: torch.Tensor) -> torch.Tensor:
        X_flat = X.reshape(X.shape[0], -1)
        X_pca = super().transform(X_flat)
        X_recon = super().inverse_transform(X_pca)
        return torch.tensor(X_recon).float().to(X.device)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        X_flat = X.reshape(X.shape[0], -1)
        X_recon = self.reconstruct(X)
        mse = torch.mean((X_flat - X_recon) ** 2, dim=1)
        return mse
