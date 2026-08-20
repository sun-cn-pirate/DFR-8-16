# DFR：MVTec AD 全 15 类复现

本仓库是在 [YoungGod/DFR](https://github.com/YoungGod/DFR) 基础上完成的
DFR（Deep Feature Reconstruction）复现工程。上游代码导入自提交
`f2e2d4ef5e542fb99aa41566cd9f662bec9ce771`，本仓库增加了现代
PyTorch 兼容、统一命令行入口、数据校验、断点续训、论文指标评估、报告生成和
15 类自动运行脚本。

## 复现状态

**已经完成 MVTec AD 全部 15 类、每类 700 epochs 的训练和测试。**

原论文第 IV-A.5 节只采用两项正式评价指标：**像素级 ROC-AUC** 和
**PRO-AUC（只积分到 30% FPR）**。因此本 README 只用这两项指标判断是否复现，
不把后来由评估程序计算的 AP、图像级分数或 Best IoU 当作论文指标。

| 配置 | 特征范围 | 像素级 ROC-AUC | PRO-AUC |
|---|---|---:|---:|
| **本次复现** | **VGG19 `f{1:12}`** | **0.94746** | **0.89045** |
| 原论文 | VGG19 `f{1:12}` | 0.94 | 0.90 |
| 原论文 | VGG19 `f{1:16}` | 0.95 | 0.91 |

## 与原论文结果对比

原论文表 II、III 同时报告了 `f{1:12}` 和 `f{1:16}`。下表把二者都列出；
本次实验证明的是 **12层复现结果**，所以与论文12层属于同配置比较，论文16层只作为
更强的参考基准，不能写成“已经复现16层”。所有数值均采用 `0–1` 标度。

| 类别 | 本次 ROC-AUC `f{1:12}` | 论文 ROC-AUC `f{1:12}` | 论文 ROC-AUC `f{1:16}` | 本次 PRO-AUC `f{1:12}` | 论文 PRO-AUC `f{1:12}` | 论文 PRO-AUC `f{1:16}` |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 0.95891 | 0.95 | 0.97 | 0.91421 | 0.92 | 0.93 |
| cable | 0.90749 | 0.88 | 0.92 | 0.82036 | 0.77 | 0.81 |
| capsule | 0.98286 | 0.98 | 0.99 | 0.94838 | 0.96 | 0.97 |
| hazelnut | 0.98599 | 0.98 | 0.99 | 0.97091 | 0.97 | 0.97 |
| metal_nut | 0.91559 | 0.90 | 0.93 | 0.84918 | 0.87 | 0.90 |
| pill | 0.96178 | 0.96 | 0.97 | 0.95168 | 0.96 | 0.96 |
| screw | 0.98565 | 0.99 | 0.99 | 0.93833 | 0.95 | 0.96 |
| toothbrush | 0.98643 | 0.98 | 0.99 | 0.94230 | 0.93 | 0.93 |
| transistor | 0.78250 | 0.75 | 0.80 | 0.62763 | 0.77 | 0.79 |
| zipper | 0.96743 | 0.96 | 0.96 | 0.88580 | 0.89 | 0.90 |
| carpet | 0.97872 | 0.96 | 0.97 | 0.94106 | 0.93 | 0.93 |
| grid | 0.97013 | 0.98 | 0.98 | 0.90134 | 0.93 | 0.93 |
| leather | 0.98983 | 0.99 | 0.98 | 0.97902 | 0.97 | 0.97 |
| tile | 0.89082 | 0.86 | 0.87 | 0.77354 | 0.79 | 0.79 |
| wood | 0.94773 | 0.94 | 0.93 | 0.91294 | 0.93 | 0.91 |
| **15类平均（本次/论文表）** | **0.94746** | **0.94** | **0.95** | **0.89045** | **0.90** | **0.91** |

结论：本次12层复现的平均像素 ROC-AUC 接近论文16层结果，并高于论文12层表中
保留到两位小数的均值；平均 PRO-AUC 略低于论文12层和16层。主要偏差来自
`transistor`：本次 PRO-AUC 为 `0.62763`，论文12层和16层分别为 `0.77`、`0.79`。
这里不使用论文只保留两位小数的数值推导五位小数的“精确差值”。

### 为什么本次跑的是12层

本次复现开始时确定的实验计划明确指定“VGG19 前12个卷积层”，上游代码的实际
运行配置也固定使用 `relu1_1` 至 `relu4_4`，对应论文的 `f{1:12}`。因此现有15套
checkpoint 文件名均包含 `l12`，它们不是16层模型。

论文结论确实是完整 `f{1:16}` 的总体均值更高。若要严格复现16层，需要加入
`relu5_1` 至 `relu5_4`，重新为每个类别估计 PCA 维度并训练15套新的 CAE；不能把
现有12层权重直接当成16层结果。

完整数值请查看：

- [逐类别原始评估输出](reports/dfr_mvtec_summary.md)（其中额外诊断字段不是论文指标）
- [机器可读 CSV 原始数据](reports/dfr_mvtec_summary.csv)
- [运行环境记录](reports/environment.json)
- [MVTec AD 数据校验记录](reports/mvtec_validation.json)

## 结果放在哪里

GitHub 上已经提交的是可复现代码、紧凑指标报告和精选异常图：

| 内容 | 仓库位置 |
|---|---|
| 15 类汇总指标 | `reports/dfr_mvtec_summary.md`、`reports/dfr_mvtec_summary.csv` |
| 15 类精选异常图 | `docs/assets/visual_examples/` |
| 代码阅读指南 | `docs/dfr_code_walkthrough.md` |
| 数据和环境校验 | `reports/mvtec_validation.json`、`reports/environment.json` |

下面是每个类别各选一张组成的结果总览：

![DFR MVTec AD 15 类精选结果](docs/assets/visual_examples/mvtec_visual_examples.jpg)

服务器上的完整结果保留在 `/root/DFR/outputs/`，由于权重和全量图片体积较大，
没有提交到 Git：

| 本地内容 | 本地位置 |
|---|---|
| 每类 CAE 权重与 PCA 维度 | `outputs/models/<类别>/.../model/` |
| 损失曲线和逐阈值指标 | `outputs/models/<类别>/.../eval/` |
| 异常分数图、预测图、二值图和 mask | `outputs/Results/<类别>/.../` |
| 全量运行日志 | `outputs/logs/`、`outputs/full-reproduction.log` |
| MVTec AD 数据集 | `/root/DFR/data/mvtec_ad` |

例如 bottle 的最终 CAE 参数位于：

```text
outputs/models/bottle/AnoSegDFR(BN)_vgg19_l12_d197_s4_k4_nearest/model/autoencoder.pth
```

`data/`、`outputs/`、预训练权重、缓存和虚拟环境由 `.gitignore` 排除。
这是有意的：GitHub 保存代码与紧凑报告，服务器保存可重新生成的大型实验产物。

## 复现配置

本次实验保持论文核心配置：

- 输入图像：256 × 256
- 特征提取器：预训练 VGG19 前 12 个卷积层（冻结，不参与训练）
- 区域特征：4 × 4 聚合核，步幅 4，最近邻特征对齐
- 特征降维：PCA 保留 90% 方差，每个类别独立确定维度
- 训练对象：每个类别一个独立 CAE
- 优化器：Adam，学习率 `1e-4`
- batch size：4
- 训练轮数：每类 700 epochs
- 训练数据：只使用各类别的 `train/good` 正常图像
- 测试标签和像素 mask：只用于测试阶段计算论文的 ROC-AUC 和 PRO-AUC

整体流程为：

```text
图像 -> VGG19 多层特征 -> 4×4 区域聚合 -> CAE 特征重建
     -> 重建误差异常分数 -> 像素级异常分割评估
```

更详细的代码入口、张量流和关键函数见
[DFR 代码阅读指南](docs/dfr_code_walkthrough.md)。

## 环境

已验证环境：

```text
Conda 环境：dfr
Python：3.12.11
PyTorch：2.13.0+cu132
torchvision：0.28.0+cu132
CUDA：13.2
GPU：NVIDIA GeForce RTX 4090
```

复用服务器已有的 CUDA PyTorch 环境并安装其余依赖：

```bash
conda create -n dfr --clone py312 -y
conda run -n dfr python -m pip install -i https://pypi.org/simple -r requirements.txt
```

本项目的安装、校验、测试和训练命令均在 `dfr` Conda 环境中运行；项目内已有
`.venv` 不使用，也不与 `py312` 混用。

## 数据集

从 [MVTec AD 官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
下载数据，并将 15 个类别目录放到 `data/mvtec_ad`。如果官方下载页面不可用，
可使用项目提供的研究用途下载脚本：

```bash
conda run -n dfr python scripts/download_mvtec.py \
  --output data/mvtec_ad \
  --workers 32
```

运行前校验类别、图片数量和 mask 对应关系：

```bash
conda run -n dfr python scripts/validate_mvtec.py data/mvtec_ad \
  --json reports/mvtec_validation.json
```

本次数据校验结果为 15 个类别、5,354 张图像和 1,258 张异常 mask。
MVTec AD 数据不随本仓库分发，仅用于遵守
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
的非商业研究。

## 运行方法

先进行全 15 类 1 epoch 冒烟测试：

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

按论文配置运行 700 epochs：

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

服务器长时间运行可使用按类别串行、支持断点恢复的脚本：

```bash
mkdir -p outputs
nohup conda run --no-capture-output -n dfr python \
  scripts/run_full_reproduction.py \
  > outputs/full-reproduction.log 2>&1 &
```

再次执行同一命令会从未完成类别的 checkpoint 恢复。若不希望脚本自动提交紧凑
报告，可增加 `--no-push`。

## 测试

```bash
conda run -n dfr python -m unittest discover -s tests
```

当前测试结果：8/8 通过。

## 分支和版本

- `main`：已完成的复现版本
- `reproduction`：复现过程分支，与当前 `main` 同步
- `dfr-reproduction-v1`：全 15 类 700 epochs 完成时创建的版本标签

## 上游代码与许可说明

原始论文代码来自 [YoungGod/DFR](https://github.com/YoungGod/DFR)。上游仓库
没有提供软件 LICENSE，因此上游源码和图片的权利仍归原作者所有；本仓库不声称
能够对其重新许可。本仓库中的兼容与复现修改也不改变这一事实。

论文：

- Yong Shi, Jie Yang, Zhiquan Qi. *Unsupervised anomaly segmentation via
  deep feature reconstruction*. Neurocomputing, 2021.
  [论文页面](https://www.sciencedirect.com/science/article/pii/S0925231220317951) ·
  [arXiv](https://arxiv.org/abs/2012.07122)

上游仓库给出的定性结果：

![上游论文定性结果](figs/seg-quality-l12.jpg)

## 引用

```bibtex
@article{DFR2020,
  title = {Unsupervised anomaly segmentation via deep feature reconstruction},
  journal = {Neurocomputing},
  year = {2020},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2020.11.018},
  url = {http://www.sciencedirect.com/science/article/pii/S0925231220317951},
  author = {Yong Shi and Jie Yang and Zhiquan Qi}
}

@misc{yang2020dfr,
  title = {DFR: Deep Feature Reconstruction for Unsupervised Anomaly Segmentation},
  author = {Jie Yang and Yong Shi and Zhiquan Qi},
  year = {2020},
  eprint = {2012.07122},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```
