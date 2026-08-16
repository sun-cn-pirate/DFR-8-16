# DFR
Project: Unsupervised Anomaly Detection and Segmentation

> This repository is a research reproduction of
> [YoungGod/DFR](https://github.com/YoungGod/DFR), imported from upstream
> commit `f2e2d4ef5e542fb99aa41566cd9f662bec9ce771`. The upstream repository
> does not provide a software license. Its source and figures remain attributed
> to the original authors; this repository does not claim to relicense them.

## Reproduction status

The reproduction targets all 15 MVTec AD categories using the paper settings:
256 x 256 images, the first 12 VGG19 convolutional feature levels, 4 x 4
regional aggregation with stride 4, PCA retaining 90% variance, batch size 4,
Adam with a learning rate of `1e-4`, and 700 training epochs per category.

MVTec AD is not included in this repository. It is used only for non-commercial
research under the
[CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Downloaded data, pretrained weights, checkpoints, and large generated artifacts
are intentionally excluded from Git.

The compatibility work and reproducible commands are developed on the
`reproduction` branch. Exact setup, validation, smoke-test, and full-run
instructions are provided below.

## Environment

The tested host has an NVIDIA RTX 4090, CUDA 13.2, Python 3.12.11, PyTorch
2.13.0+cu132, and torchvision 0.28.0+cu132. Clone the host's `py312` Conda
environment so the CUDA-enabled PyTorch installation is reused, then install
the remaining packages directly from the official Python Package Index:

```bash
conda create -n dfr --clone py312 -y
conda run -n dfr python -m pip install -i https://pypi.org/simple -r requirements.txt
```

All setup, validation, testing, and training commands below run exclusively in
the `dfr` environment. The pre-existing project `.venv` is not used.

## Dataset

Download MVTec AD from the [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
and extract the 15 category folders under `data/mvtec_ad`. If the official form
is unavailable, the approved research fallback can be downloaded with:

```bash
conda run -n dfr python scripts/download_mvtec.py \
  --output data/mvtec_ad \
  --workers 32
```

Always validate the directory structure, published image counts, and masks
before running an experiment:

```bash
conda run -n dfr python scripts/validate_mvtec.py data/mvtec_ad \
  --json reports/mvtec_validation.json
```

Neither the dataset nor generated model weights are tracked by Git.

## Run the reproduction

One-epoch smoke test across all 15 categories:

```bash
conda run -n dfr python DFR-source/main.py \
  --mode all \
  --data-root data/mvtec_ad \
  --categories all \
  --epochs 1 \
  --checkpoint-every 1 \
  --metric-steps 100 \
  --no-save-visualizations \
  --resume
```

Paper-configuration run (700 epochs per category):

```bash
conda run -n dfr python DFR-source/main.py \
  --mode all \
  --data-root data/mvtec_ad \
  --categories all \
  --epochs 700 \
  --checkpoint-every 10 \
  --metric-steps 5000 \
  --resume
```

For a long-running server experiment, use the resumable serial runner. It
validates the complete dataset first, writes a persistent per-category log
under `outputs/logs/`, checks every metric for finite values, and commits and
pushes only the compact reports after each successful category:

```bash
mkdir -p outputs
nohup conda run --no-capture-output -n dfr python \
  scripts/run_full_reproduction.py \
  > outputs/full-reproduction.log 2>&1 &
```

Rerunning the same command resumes the saved epoch for the interrupted
category. Pass `--no-push` when publication is not wanted.

The command writes checkpoints and large visual artifacts below `outputs/`.
Compact cross-category results are updated in `reports/dfr_mvtec_summary.csv`
and `reports/dfr_mvtec_summary.md` after each completed category.

Paper: Unsupervised anomaly segmentation via deep feature reconstruction  | **[Neurocomputing]**[`pdf`](https://www.sciencedirect.com/science/article/pii/S0925231220317951)[`code`](https://github.com/YoungGod/DFR) | **arxive preprint**[`pdf`](https://arxiv.org/abs/2012.07122)

Introduction: Automatic detecting anomalous regions in images of objects or textures without priors of the anomalies is challenging, especially when the anomalies appear in very small areas of the images, making difficult-to-detect visual variations, such as defects on manufacturing products.
	This paper proposes an effective unsupervised anomaly segmentation approach that can detect and segment out the anomalies in small and confined regions of images. Concretely, we develop a multi-scale regional feature generator which can generate multiple spatial context-aware representations from pre-trained deep convolutional networks for every subregion of an image. 
	The regional representations not only describe the local characteristics of corresponding regions but also encode their multiple spatial context information, making them discriminative and very beneficial for anomaly detection.
	Leveraging these descriptive regional features, we then design a deep yet efficient convolutional autoencoder and detect anomalous regions within images via fast feature reconstruction.
	Our method is simple yet effective and efficient. It advances the state-of-the-art performances on several benchmark datasets and shows great potential for real applications.
	
# Qualitative results
![image](https://github.com/YoungGod/DFR/tree/master/figs/seg-quality-l12.jpg)

# Citation
If you find something useful, wellcome to cite our paper:
```
@article{YANG2022108874,
title = {Learning Deep Feature Correspondence for Unsupervised Anomaly Detection and Segmentation},
journal = {Pattern Recognition},
pages = {108874},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.108874},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322003557},
author = {Jie Yang and Yong Shi and Zhiquan Qi},
}
```
```
@article{DFR2020,
    title = "Unsupervised anomaly segmentation via deep feature reconstruction",
    journal = "Neurocomputing",
    year = "2020",
    issn = "0925-2312",
    doi = "https://doi.org/10.1016/j.neucom.2020.11.018",
    url = "http://www.sciencedirect.com/science/article/pii/S0925231220317951",
    author = "Yong Shi and Jie Yang and Zhiquan Qi",
}
```

```
@misc{yang2020dfr,
      title={DFR: Deep Feature Reconstruction for Unsupervised Anomaly Segmentation}, 
      author={Jie Yang and Yong Shi and Zhiquan Qi},
      year={2020},
      eprint={2012.07122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
