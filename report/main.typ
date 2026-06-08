#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Anomaly Detection in Low-Resolution Images Using Convolutional Autoencoders with Hybrid Scoring and Test-Time Augmentation],
  abstract: [
    Anomaly detection in low-resolution images is a fundamental computer vision task requiring a careful balance between sensitivity and specificity. In this study, we #footnote[The first-person plural is used here as a pluralis modestiae, not a pluralis maiestatis] develop and evaluate an anomaly detection system using the MNIST dataset, explicitly reserving digit '8' as a validation anomaly and digit '4' as an unseen test anomaly. We compare a linear Principal Component Analysis (PCA) baseline against a non-linear Convolutional Autoencoder (AE). To further enhance the AE's performance, we propose and evaluate three experimental modifications: a Hybrid Anomaly Score combining Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM), a self-supervised denoising training, and a Test-Time Augmentation (TTA) inference strategy that measures the structural instability of reconstructions across micro-rotations. Our results demonstrate that the AE significantly outperforms the PCA baseline. Additionally, the integration of the Hybrid Loss and TTA variance scoring results in the highest predictive confidence, successfully exploiting the network's spatial biases to isolate anomalous data.
  ],
  authors: (
    (
      name: "github.com/slymachenko",
    ),
  ),
  index-terms: ("Anomaly detection", "Autoencoder", "PCA", "MNIST", "Test-Time Augmentation", "SSIM"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
Anomaly detection is one of many important applications of Machine Learning, enabling the efficient detection of outliers in large datasets. A robust anomaly detection system must extract the core structural manifold of normal data while balancing the trade-off between false alarms (false positives) and missed anomalies (false negatives).

Our goal in this study is to build a high-quality, efficient anomaly detection system built for low-resolution images. We utilize the MNIST dataset @lecun1998gradient, reformulating it into a one-class classification problem. We utilize the MNIST dataset, partitioning the classes to simulate an anomaly detection scenario. Digits 8 and 4 are isolated entirely from the training phase. Digit 8 is utilized as a validation set for hyperparameter tuning, while digit 4 is strictly reserved for final, unseen testing.

The system is trained exclusively on normal data to perform image reconstruction. The core assumption is that the reconstruction error (RE) acts as a proxy for the anomaly score: normal data will result in a low RE as it falls within the learned manifold, whereas anomalous data will result in a higher RE. @sakurada2014anomaly.

To systematically improve the detector, we establish a linear PCA baseline and a non-linear Convolutional Autoencoder. Afterwards, we perform a series of targeted experiments on the Autoencoder: latent dimensionality fine-tuning, anomaly score formulation using Hybrid Loss, self-supervised denoising, and input transformation ensembles via multi-rotation variance.

= Exploratory Data Analysis

Before defining the architectures, we perform dimensionality reduction to understand the natural clustering and variance of the dataset.

#figure(
  placement: top,
  image("../assets/figs/exploration/tSNE.png"),
  caption: [Mean image and the top 5 PCA basis images (eigendigits), illustrating the primary structural variations captured by the linear components.],
) <fig:tSNE>

We can see in @fig:tSNE that 2D t-SNE embedding @vandermaaten2008visualizing of the MNIST dataset reveals that digits '4' and '9' form closely adjacent clusters, suggesting that '4' may be particularly difficult to isolate if '9' is defined as normal data.

Additionally, we apply Principal Component Analysis (PCA) to extract the eigenvectors of the dataset's covariance matrix. Analysis shows a logarithmic trend in explained variance, where 25 components capture approximately 70% of the data's variance.

#figure(
  placement: top,
  image("../assets/figs/exploration/mean_and_top10_eigendigits.png"),
  caption: [Mean image and the top 5 PCA basis images (eigendigits), illustrating the primary structural variations captured by the linear components.],
) <fig:pca_components>

#figure(
  placement: top,
  image("../assets/figs/exploration/pca_extreme_samples_pc1_pc2_overlays.png"),
  caption: [Extreme samples projected onto PC1 and PC2, overlaid with their respective basis masks, demonstrating how PCA captures linear spatial dependencies.],
) <fig:pca_extreme>

As seen in @fig:pca_components and @fig:pca_extreme, the principal components encode combinations of comprehensive structural traits (e.g., PC1 separates straight vertical strokes from rounded loops). This suggests that a linear model can capture basic geometric features, forming a viable baseline.

= Methods <sec:methods>

== Metric Selection
For hyperparameter optimization across all models, Precision-Recall Area Under the Curve (PR AUC) is selected as the primary maximization objective. Under heavy class imbalance, ROC AUC can provide an overly optimistic view of performance @saito2015precision. PR AUC is highly sensitive to false positives, making it a better metric for evaluating anomaly detection systems.

== PCA Baseline Detection
We implement a PCA-based anomaly detector to serve as the baseline. The optimal number of components is determined via a zooming grid search over the validation set, resulting in an optimal configuration of 24 components. While the PCA system demonstrates high recall, it struggles with spatial dependencies, leading to a high rate of false positives and a generally blurred reconstruction of anomalies.

== Autoencoder Architecture
To overcome the spatial limitations of PCA, we design a Convolutional Autoencoder (AE). The architecture utilizes a deep bottleneck structure:
- Encoder: Two `Conv2d` layers (1 -> 16 -> 32 channels) with 3x3 kernels, stride 2, and padding 1, followed by a flattened linear projection into the latent space. `LeakyReLU` activation is used to prevent dying gradients @maas2013rectifier.
- Decoder: A symmetric expansion using linear upsampling and `ConvTranspose2d` layers, ending in a `Sigmoid` activation to output pixel intensities in the $[0, 1]$ range.

The latent dimensionality is heavily constrained to force the network to learn a compressed semantic representation rather than a trivial identity mapping.

= Experiments <sec:experiments>

We conduct a series of experiments targeting distinct stages of the Autoencoder pipeline:

== Base Optimization

We optimize the latent dimensionality of the Autoencoder using a grid search from 2 to 16 dimensions.

#figure(
  placement: top,
  image("../assets/figs/autoencoder/ae_lat_dim_vs_prauc.png"),
  caption: [Latent Dimension Optimization. A bottleneck of 9 dimensions maximizes the validation PR AUC.],
) <fig:ae_lat_dim>

As shown in @fig:ae_lat_dim, a latent dimension of 9 yields the highest PR AUC ($~0.825$). This model establishes our foundational Autoencoder performance.

== Anomaly Score Formulation
Standard models utilize pixel-wise Mean Squared Error (MSE) as the anomaly measure. However, MSE fails to account for perceptual structures. We propose incorporating the Structural Similarity Index Measure (SSIM) @wang2004image, which evaluates the visual impact of shifts in image luminance ($mu$), contrast ($sigma$), and structure ($sigma_{x y}$):

$
  "SSIM"(x, y) = ((2 mu_x mu_y + c_1)(2 sigma_{x y} + c_2)) / ((mu_x^2 + mu_y^2 + c_1)(sigma_x^2 + sigma_y^2 + c_2))
$ <eq:ssim>

We combine MSE and SSIM into a hybrid anomaly score ($"AS"$) governed by a blending parameter $alpha$. Because SSIM outputs a similarity score between $-1$ and $1$, we apply the transformation $(1 - "SSIM") / 2$ to convert it into a bounded loss ranging from $0$ to $1$:

$ "AS"(alpha) = alpha "MSE" + (1 - alpha) (1 - "SSIM") / 2 $ <eq:hybrid_loss>

Through optimization, we find $alpha = 0.81$ provides the best performance, allowing the model to leverage both pixel-level accuracy and structural clarity @bergmann2018improving.

== Training Logic (Self-Supervised Masking)

We hypothesize that training the Autoencoder as a Denoising Autoencoder (DAE) @vincent2008extracting will force a more robust manifold representation. We inject Gaussian noise parameterized by standard deviation $sigma$ during training.

Interestingly, the hyperparameter optimization collapsed to $sigma = 0.0$. Even minor noise regularizations degraded the sharp reconstruction boundaries required to maximize the PR AUC gap between normal and anomalous samples.

== Inference Logic (Test-Time Augmentation)

Since the Autoencoder is trained strictly on upright digits, it is not rotation-invariant. We hypothesize that normal digits retain enough structural similarity during micro-rotations to result in consistent, low-variance reconstructions. In contrast, anomalous digits --- already poorly mapped in the latent space --- will suffer from extreme structural instability across rotations.

Hence, we propose a Test-Time Augmentation (TTA) ensemble via multi-rotation variance. We rotate a single input image by multiple micro-angles ($-10^degree$, $0^degree$, $10^degree$) and pass them through the network.

To accurately measure this instability without introducing spatial misalignment penalties, reconstructions are rotated back to $0^degree$ before calculating the pairwise $"AS"$ variance across the ensemble. This variance is then blended with the base reconstruction score via a weighting parameter $lambda$. We run a zooming grid search between $0.0$ and $10.0$ to optimize this penalty, finding an optimal value of $lambda = 4.6$, confirming that variance scaling heavily contributes to anomaly separation.

= Results <sec:results>

The qualitative and quantitative results confirm the superiority of the optimized Autoencoder over the PCA baseline.

== PCA Baseline Performance
To establish a performance baseline, we evaluate the linear Principal Component Analysis (PCA) anomaly detector configured with an optimal 24 components. The anomaly score for the PCA model is strictly derived from the pixel-wise Mean Squared Error (MSE) between the input image and its reconstruction.

#figure(
  placement: bottom,
  grid(
    image("../assets/figs/pca_baseline/pca_anomaly_score_distribution.png"),
    image("../assets/figs/pca_baseline/pca_performance_curves.png"),
  ),
  caption: [PCA Anomaly Score Distribution (top) and Performance Curves (bottom). The high degree of overlap between normal and anomalous distributions severely limits the model's precision.],
) <fig:pca_results>

As illustrated in @fig:pca_results, the linear PCA framework struggles to cleanly separate the normal and anomalous data distributions. While the model achieves a high recall (identifying most anomalies), it suffers from exceptionally low precision (0.3662). This results in an overwhelming number of false positives (1502), heavily depressing the PR AUC score to 0.4916. The fundamental limitation lies in the PCA's reliance on rigid, global linear combinations, which prevents it from capturing the nuanced spatial hierarchies required to accurately reconstruct and isolate complex out-of-distribution structures.

#figure(
  placement: none,
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../assets/figs/pca_baseline/pca_rec_val_normal.png"),
    image("../assets/figs/pca_baseline/pca_rec_val_anomaly.png"),
  ),
  caption: [PCA Reconstructions for Normal Data (left) and Anomalous Data (right).],
) <fig:pca_recons>

Qualitative analysis of the PCA reconstructions (@fig:pca_recons) confirms this limitation. The linear basis vectors output highly blurred, "average" geometric approximations for both normal and anomalous digits. Because the PCA manifold cannot effectively model the sharp, non-linear edge dependencies inherent in the MNIST dataset, the reconstruction error for normal digits remains artificially high, ultimately bridging the scoring gap between the two classes.

== Autoencoder Performance

#figure(
  placement: none,
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../assets/figs/autoencoder/ae_rec_val_normal.png"),
    image("../assets/figs/autoencoder/ae_rec_val_anomaly.png"),
  ),
  caption: [Autoencoder reconstructions for normal (left) and anomalous (right) validation data. Note how the anomaly '8' is warped into a generic '0' or '3' shape, yielding a high anomaly score.],
) <fig:ae_recons>

As seen in @fig:ae_recons, the Autoencoder successfully reconstructs normal digits but heavily distorts the unseen anomalous digits by projecting them onto the normal data manifold.

Furthermore, the TTA variance hypothesis proved itself to be effective.

#figure(
  placement: none,
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../assets/figs/ae_experiments/exp3_rec_val_normal_rotations.png"),
    image("../assets/figs/ae_experiments/exp3_rec_val_anomaly_rotations.png"),
  ),
  caption: [Micro-rotations (top row) and their aligned reconstructions (bottom row). Left: Normal digits remain structurally stable. Right: Anomalous digits exhibit severe structural hallucination and variance.],
) <fig:tta_qualitative>

@fig:tta_qualitative illustrates the exact mechanism of the TTA ensemble. The normal digit maintains its form, resulting in low pairwise variance. The anomalous digit wildly fluctuates in structure across rotations. When incorporated into the final scoring metric ($lambda = 4.6$), the TTA ensemble achieves the highest overall performance on the test set.

= Final Evaluation

Using the fully optimized parameters ($dim=9$, $alpha=0.81$, $lambda=4.6$), we evaluate the final model against the completely unseen test set (anomaly digit '4').

#figure(
  placement: none,
  grid(
    image("../assets/figs/evaluation/eval_anomaly_score_distribution.png"),
    image("../assets/figs/evaluation/eval_performance_curves.png"),
  ),
  caption: [Final Autoencoder Anomaly Score Distribution (top) and Performance Curves (bottom).],
) <fig:ae_results>

The model demonstrates excellent generalizability, achieving an F2-score of 0.7925 and a PR AUC of 0.7386 on entirely unseen anomalies (@fig:ae_results). The threshold securely separates the majority of the distributions.

#figure(
  placement: none,
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("../assets/figs/evaluation/test_rec_test_normal.png"),
    image("../assets/figs/evaluation/test_rec_test_anomaly.png"),
  ),
  caption: [Final Autoencoder reconstructions for normal test data (left) and unseen anomalous test data (right).],
) <fig:test_recons>

Qualitative analysis of the test reconstructions (@fig:test_recons) confirms our initial finding. When the Autoencoder attempts to reconstruct the anomalous digit 4, it consistently hallucinates a digit 9. Because 4 and 9 share a heavily overlapping latent cluster (as observed in the initial t-SNE embedding), the network maps the unknown '4' onto the nearest known geometric manifold, which happens to be a '9'. This geometric forcing generates the high structural error which is used to accurately flag the image as an anomaly.

= Discussion <sec:discussion>
The experiments confirm that the predictive power of an Autoencoder-based anomaly detector lies in its controlled failures. Restricting latent dimensionality and penalizing structural deviations via Hybrid Loss forces the model to fail predictably on anomalous data. Furthermore, the Test-Time Augmentation (TTA) variance ensemble effectively exploits the network's lack of rotational invariance. The spatial strictness of the Autoencoder acts as an anomaly amplifier: in-distribution data bends but recovers, while out-of-distribution data fails to recover, causing variance spikes that separate the two classes more easily.

However, the current solution faces limitations. The TTA ensemble significantly increases inference latency by requiring multiple forward passes per image, scaling poorly for real-time applications. The deep Convolutional Autoencoder architecture may also be overly complex for 28x28 grayscale images, leading to suboptimal energy and memory use.

Future improvements should focus on efficiency and architectural refinement to approach production readiness. Simplifying the network through shallower architecture or quantization could drastically cut inference time preserving the bottleneck constraints. To reduce TTA latency, ensemble generation could be parallelized.

= Conclusions <sec:conclusions>
In this report, we demonstrated the development of an anomaly detection system for low-resolution images. The non-linear Convolutional Autoencoder significantly outperformed the linear PCA baseline. By defining a Hybrid Anomaly Score utilizing SSIM, and deploying a Test-Time Augmentation ensemble measuring pairwise reconstruction variance, we achieved highly robust separation of unseen anomalies. This architecture proves that exploiting a network's structural biases and inference-time instability is a highly effective strategy for anomaly detection.
