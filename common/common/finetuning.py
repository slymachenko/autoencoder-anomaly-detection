import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from typing import Callable, Tuple, Dict, Any, List

def optimize_via_zooming_grid(
    score_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    param_min: float,
    param_max: float,
    size: int = 4,
    depth: int = 1,
    param_name: str = "Param",
    is_integer: bool = False,
    verbose: bool = False
) -> Tuple[float, float, Dict[str, List[Any]]]:
    """Finds the hyperparameter value that maximizes PR AUC."""
    current_min = param_min
    current_max = param_max
    global_best_auc = -1.0
    global_best_param = None
    history = {
        "zoom_step": [],
        param_name: [],
        "auc": []
    }

    for step in range(depth):
        if verbose:
            print(f"\nZoom Step {step+1}/{depth} Interval: [{current_min:.4f}, {current_max:.4f}]")
            print(f"{param_name:<10} | {'PR AUC':<10}")

        grid = np.linspace(current_min, current_max, size)
        if is_integer:
            grid = np.unique(np.round(grid).astype(int))
        local_best_auc = -1.0
        local_best_param = None

        for param_val in grid:
            param_val = int(param_val) if is_integer else float(param_val)
            val_normal_scores, val_anomaly_scores = score_fn(param_val)

            y_true = np.concatenate([np.zeros_like(val_normal_scores, dtype=int), np.ones_like(val_anomaly_scores, dtype=int)])
            y_scores = np.concatenate([val_normal_scores, val_anomaly_scores])

            auc = average_precision_score(y_true, y_scores)

            if verbose:
                print(f"{param_val:<10.4f} | {auc:<10.4f}")

            history["zoom_step"].append(step + 1)
            history[param_name].append(param_val)
            history["auc"].append(float(auc))

            if auc > local_best_auc:
                local_best_auc = auc
                local_best_param = param_val
            if auc > global_best_auc:
                global_best_auc = auc
                global_best_param = param_val

        # Update bounds around the local best configuration found in this step
        step_size = (current_max - current_min) / (size - 1)
        current_min = np.clip(local_best_param - (step_size / 2), param_min, param_max)
        current_max = np.clip(local_best_param + (step_size / 2), param_min, param_max)

        if is_integer:
            current_min, current_max = np.floor(current_min), np.ceil(current_max)

    return global_best_param, global_best_auc, history

def optimize_fbeta_threshold(
    val_normal_scores: np.ndarray, 
    val_anomaly_scores: np.ndarray, 
    beta: float = 2.0
) -> Tuple[float, Dict[str, Any]]:
    """Calculates the exact operational threshold maximizing F-beta."""
    y_true = np.concatenate([np.zeros_like(val_normal_scores), np.ones_like(val_anomaly_scores)])
    y_scores = np.concatenate([val_normal_scores, val_anomaly_scores])

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    precision, recall, thresholds = precision[:-1], recall[:-1], pr_thresholds

    fbeta_vals = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-12)
    best_idx = np.nanargmax(fbeta_vals)

    results = {
        'y_true': y_true,
        'y_scores': y_scores,
        'thresholds': thresholds,
        'fbeta_vals': fbeta_vals,
        'precision': precision[best_idx],
        'recall': recall[best_idx],
        'fbeta': fbeta_vals[best_idx]
    }
    return thresholds[best_idx], results