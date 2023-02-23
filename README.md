This project investigated the research question of whether incorporating **feedback connections** improves performance on the **Robust Detection Benchmark**.

The following two references served as the main foundation:

> Jarvers, C. and H. Neumann (2019) Incorporating feedback in convolutional neural net-
works, paper presented at the Proceedings of the Cognitive Computational Neuroscience
Conference, 395–398.

> Michaelis, C., B. Mitzkus, R. Geirhos, E. Rusak, O. Bringmann, A. S. Ecker, M. Bethge
and W. Brendel (2019) Benchmarking robustness in object detection: Autonomous
driving when winter is coming, https://arxiv.org/abs/1907.07484.

----------------------------------------------------------------
The repository is a fork of the [OpenMMLab Detection Toolbox](https://github.com/open-mmlab/mmdetection) which is based on PyTorch.

## Installation

Please refer to the [Installation instructions](docs/en/get_started.md) from the MMdetection docs.

## Results Robust Detection Benchmark
| model | mAP |  mPC   |  rPC |
|---|--|---|---|
|FF-Baseline | 0.801 | 0.489 | 60,5 % |
|FB-Mod | 0.806 | 0.468 | 57,5 % |
|FB-Add | 0.803 | 0.457 | 56,3 % |

[mAP] Performance on Clean Data in AP50  
[mPC] Mean Performance under Corruption in AP50  
[rPC] Relative Performance under Corruption in % 