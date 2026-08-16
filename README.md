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
`reproduction` branch. Exact setup and execution instructions will be added as
the smoke tests are validated.

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
