# Anomaly Detection in Low-Resolution Images Using Convolutional Autoencoders with Hybrid Scoring and Test-Time Augmentation

This repository contains the codebase and raw scientific report for an **Anomaly Detection System** designed for low-resolution images. The project compares a linear Principal Component Analysis (PCA) baseline against a non-linear Convolutional Autoencoder (AE), evaluating advanced techniques such as Hybrid Loss (MSE + SSIM) and Test-Time Augmentation (TTA) variance scoring.

## Setup

> [!NOTE]
> You must have [Git](https://git-scm.com/downloads) and [Conda](https://www.anaconda.com/download/success) installed on your system

1. Clone the repository:

    `git clone https://github.com/slymachenko/autoencoder-anomaly-detection.git`

2. Navigate to the project directory.
3. Create conda environment:

    ```bash
    conda env create -f environment.yml
    conda activate aad
    ```

## Dataset Formulation

We utilize the MNIST dataset with a highly specific evaluation split:

- Normal Data: Digits `0, 1, 2, 3, 5, 6, 7, 9`
- Validation Anomaly: Digit `8` (Used strictly for hyperparameter tuning, threshold calibration, and validation).
- Test Anomaly: Digit `4` (Held out entirely to ensure an unbiased evaluation of the final optimized system).

*Note: Initial t-SNE exploratory data analysis revealed that digits `4` and `9` share a heavily overlapping latent cluster, making `4` an exceptionally challenging unseen anomaly to detect.*

## Methodology & Experiments

The system is evaluated using **Precision-Recall Area Under the Curve (PR AUC)** and the **F2-Score** to prioritize recall and handle class imbalances.

1. **PCA Baseline:** A linear baseline utilizing 24 principal components. While achieving high recall, it struggles with non-linear spatial dependencies, resulting in higher false positive rates.
2. **Convolutional Autoencoder (AE):** A deep bottleneck architecture (tuned to an optimal latent dimension of 9) that significantly outperforms the PCA baseline.

### Autoencoder Enhancements

We conducted three targeted experiments to push the boundaries of the Autoencoder's predictive confidence:

- **Experiment 1 (Hybrid Loss):** Replaced standard MSE with a Hybrid Anomaly Score combining absolute pixel fidelity (MSE) with structural coherence (Structural Similarity Index Measure - SSIM).
- **Experiment 2 (Self-Supervised Masking):** Attempted to train the model as a Denoising Autoencoder (DAE) via Gaussian noise injection.
- **Experiment 3 (Test-Time Augmentation):** Developed a novel inference strategy that measures the structural instability of reconstructions across multiple micro-rotations (-10deg to +40deg). Reconstructions are realigned to 0deg, and their pairwise variance is penalized heavily to isolate anomalies.

## Key Findings

The TTA ensemble (Exp 3) proved to be effective. Because the network is not rotation-invariant, normal digits bend but recover structurally during micro-rotations. Unseen anomalous digits, however, fail to recover, causing massive variance spikes that easily separate the classes.

During the final test evaluation, the Autoencoder consistently hallucinates a digit `9` when attempting to reconstruct the unseen anomalous digit `4`. This geometric forcing generates the massive structural error required to accurately flag the image as an anomaly, perfectly confirming our initial t-SNE manifold finding.
