# DFR code walkthrough

This note summarizes the reproduction code after the full 15-category MVTec AD run.
It is meant as a reading map for understanding the implementation before making
method changes.

## Big picture

DFR is a one-class anomaly localization method. Training only uses normal images
from `train/good`. Test labels and masks are used later for metrics, not for
gradient updates.

The implemented pipeline is:

```text
normal image
  -> frozen VGG19 feature maps
  -> resize all selected maps to 256x256
  -> 4x4 regional averaging with stride 4
  -> concatenate multi-layer regional features
  -> estimate CAE bottleneck width with PCA(90% variance)
  -> train a 1x1 convolutional autoencoder on normal features

test image
  -> same frozen VGG19/regional feature extractor
  -> CAE reconstruction
  -> per-location reconstruction error
  -> upsampled anomaly score map
  -> image-level and pixel-level metrics
```

## Entry points

- `DFR-source/main.py` is the unified CLI.
  - It defines the 15 MVTec categories and the paper feature-layer set
    `relu1_1` through `relu4_4`.
  - `--mode train|evaluate|all` controls whether a category is trained,
    evaluated, or both.
  - `--categories all` expands to all 15 categories.
  - `write_summary()` writes the compact CSV/Markdown reports and environment
    metadata.
- `scripts/run_full_reproduction.py` is the long-run orchestrator.
  - It first runs `scripts/validate_mvtec.py`.
  - It executes `DFR-source/main.py` one category at a time with `--resume`.
  - After each category, it verifies that the report row reached the requested
    epoch count and that metrics are finite.
  - It commits and pushes only compact report files.

## Data loading

- `DFR-source/MVTec.py` contains dataset classes.
- `NormalDataset` loads only `train/good` and returns normalized RGB tensors.
  Grayscale source images are converted to RGB for VGG compatibility.
- `TestDataset` loads every image under `test`.
  - `test/good` samples get an all-zero mask.
  - Defective samples resolve masks from
    `ground_truth/<defect_type>/<image_stem>_mask.png`.
  - Missing masks raise an error instead of silently producing bad metrics.
- `scripts/validate_mvtec.py` checks the original MVTec structure, public counts,
  and one-to-one anomalous-image/mask correspondence. The verified totals are
  5,354 images and 1,258 masks.

## Feature extractor

- `DFR-source/vgg19.py` wraps pretrained VGG19 feature layers.
- `DFR-source/feature.py` builds the regional multi-scale feature tensor.
- For each selected VGG layer:
  1. resize the feature map to `256x256`;
  2. apply replication padding;
  3. aggregate local regions with a `4x4` average window and stride `4`;
  4. concatenate channels from all selected layers.
- With the paper settings, the spatial feature grid is `64x64`. The channel
  count depends on the selected VGG layers; the CAE bottleneck size is category
  dependent.

## CAE and loss

- `DFR-source/feat_cae.py` defines `FeatCAE`.
- The active model is a symmetric 1x1 convolutional autoencoder:
  - encoder: `in_channels -> mid -> 2*latent_dim -> latent_dim`;
  - decoder: `latent_dim -> 2*latent_dim -> mid -> in_channels`.
- The training loss is mean squared reconstruction error:

```python
loss = torch.mean((x - x_hat) ** 2)
```

Only the CAE parameters are optimized with Adam. VGG19 is used as a frozen
feature extractor, and PCA is used only to choose the latent dimension.

## Training and checkpointing

- `AnoSegDFR.train()` in `DFR-source/anoseg_dfr.py` is the training loop.
- Each batch does:
  1. extract normal features under `torch.no_grad()`;
  2. reconstruct them with the CAE;
  3. backpropagate MSE reconstruction loss through the CAE only.
- Checkpoints are saved every `--checkpoint-every` epochs and at the end.
- The checkpoint includes:
  - epoch;
  - CAE weights;
  - optimizer state;
  - latent dimension;
  - config;
  - seed;
  - CPU/CUDA RNG states;
  - cumulative training time.
- `--resume` restores the checkpoint and continues from the next epoch.

## Scoring and metrics

- `AnoSegDFR.score()` computes the per-pixel anomaly map:
  1. extract test features;
  2. reconstruct features with the CAE;
  3. compute channel-mean squared reconstruction error;
  4. upsample the `64x64` error map to `256x256`.
- Image-level detection uses one score per image:
  - label: whether the ground-truth mask contains any anomalous pixel;
  - prediction: maximum anomaly score in the image.
- Pixel-level segmentation uses every pixel:
  - label: flattened ground-truth mask;
  - prediction: flattened anomaly score map.
- Final reported metrics are:
  - detection AP and ROC-AUC;
  - segmentation AP and ROC-AUC;
  - PRO-AUC up to 30% FPR;
  - best per-image IoU over the scanned thresholds.

## Important interpretation notes

- AUC is not a training loss. It is computed during validation/final evaluation
  from test masks and labels.
- The experiment is unsupervised in training, but supervised in evaluation
  because MVTec provides test labels and pixel masks.
- Each MVTec category gets its own CAE and latent dimension. VGG19 and the
  hyperparameters are shared; the learned checkpoint is category-specific.
- Large artifacts remain local under `outputs/` and are ignored by Git.
  The repository stores code, configs, compact reports, and validation metadata.

## Full-run result snapshot

The completed run used Python 3.12.11, PyTorch 2.13.0+cu132, CUDA 13.2, and an
NVIDIA GeForce RTX 4090 in the `dfr` Conda environment. All 15 categories reached
700 epochs with finite metrics. The compact final means are:

- detection AP: `0.96905`
- detection ROC-AUC: `0.93585`
- pixel AP: `0.46119`
- pixel ROC-AUC: `0.94746`
- PRO-AUC: `0.89045`
- best IoU: `0.32075`
