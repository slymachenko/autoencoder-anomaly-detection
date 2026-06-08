import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from decimal import Decimal

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
    visited_params = set()

    def get_decimals(val: float) -> int:
        if isinstance(val, int):
            return 0
        d = Decimal(str(val)).normalize()
        return abs(d.as_tuple().exponent) if d.as_tuple().exponent < 0 else 0

    base_decimals = max(get_decimals(param_min), get_decimals(param_max))

    for step in range(1, depth+1):
        current_decimals = base_decimals + step

        if verbose:
            print(f"\nZoom Step {step}/{depth} Interval: [{current_min:.4f}, {current_max:.4f}]")
            print(f"{param_name:<10} | {'PR AUC':<10}")

        # make sure only unique values are in the grid
        candidates = []
        res_mul = 1

        while len(candidates) < size and res_mul <= 20:
            raw_grid = np.linspace(current_min, current_max, size * res_mul)

            if is_integer:
                raw_grid = np.round(raw_grid).astype(int)
            else:
                raw_grid = np.round(raw_grid, decimals=current_decimals)

            for val in raw_grid:
                val = int(val) if is_integer else float(val)
                # Check against history AND current candidate list
                if val not in visited_params and val not in candidates:
                    candidates.append(val)
                    if len(candidates) == size:
                        break
            res_mul += 1

        grid = sorted(candidates)

        for param_val in grid:
            param_val = int(param_val) if is_integer else float(param_val)
            val_normal_scores, val_anomaly_scores = score_fn(param_val)

            y_true = np.concatenate([np.zeros_like(val_normal_scores, dtype=int), np.ones_like(val_anomaly_scores, dtype=int)])
            y_scores = np.concatenate([val_normal_scores, val_anomaly_scores])

            auc = average_precision_score(y_true, y_scores)

            if verbose:
                print(f"{param_val:<10.4f} | {auc:<10.4f}")

            history["zoom_step"].append(step)
            history[param_name].append(param_val)
            history["auc"].append(float(auc))

            if auc > global_best_auc:
                global_best_auc = auc
                global_best_param = param_val

        # Update bounds around the local best configuration found in this step
        step_size = (current_max - current_min) / max(len(grid) - 1, 1)
        current_min = np.clip(global_best_param - step_size, param_min, param_max)
        current_max = np.clip(global_best_param + step_size, param_min, param_max)

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