from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import torch

from typing import Tuple

def load_and_split_mnist(mnist_path: str | Path, val_anomaly_digit: int, test_anomaly_digit: int, random_seed: int = 42) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Loads MNIST data from an NPZ file and splits it into training, validating, testing subsets.

    Returns:
        A tuple containing 3 pairs of (features, labels):
        (train_normal, val_normal, val_anomaly, test_normal, test_anomaly)
    """
    mnist = np.load(mnist_path)

    # Convert to PyTorch tensors
    x_train: torch.Tensor = torch.from_numpy(mnist["train_data"]).float()
    y_train: torch.Tensor = torch.from_numpy(mnist["train_labels"]).float()
    x_test: torch.Tensor = torch.from_numpy(mnist["test_data"]).float()
    y_test: torch.Tensor = torch.from_numpy(mnist["test_labels"]).float()

    # Create masks for target digits
    train_rest_mask = ~((y_train[:, val_anomaly_digit] == 1) | (y_train[:, test_anomaly_digit] == 1))
    train_normal = (x_train[train_rest_mask], y_train[train_rest_mask])

    val_anomaly_mask = y_test[:, val_anomaly_digit] == 1
    test_anomaly_mask = y_test[:, test_anomaly_digit] == 1

    val_anomaly = (x_test[val_anomaly_mask], y_test[val_anomaly_mask])
    test_anomaly = (x_test[test_anomaly_mask], y_test[test_anomaly_mask])

    test_rest_mask = ~(val_anomaly_mask | test_anomaly_mask)
    x_test_rest = x_test[test_rest_mask]
    y_test_rest = y_test[test_rest_mask]

    # Split test_normal to val/test
    x_val_norm, x_test_norm, y_val_norm, y_test_norm = train_test_split(
        x_test_rest,
        y_test_rest,
        test_size=0.5,
        random_state=random_seed
    )

    val_normal = (x_val_norm, y_val_norm)
    test_normal = (x_test_norm, y_test_norm)

    return train_normal, val_normal, val_anomaly, test_normal, test_anomaly