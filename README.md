# Feedback Networks for Robust Object Detection 
This project investigated the research question of whether incorporating **feedback connections** improves performance on the **Robust Detection Benchmark**.

## Abstract
Object detection is a common use case for deep neural networks. In order to deploy a model in a real-world application, it must be robust to a range of possible image corruptions. This project investigated feedback connections to improve the detection robustness. Two feedback networks were implemented on the basis of a Faster R-CNN architecture with a ResNet-50 backbone. While one used additive feedback, the other one employed modulation as a coupling scheme. Evaluation on the Robust Detection Benchmark revealed that the feedback networks did not improve robustness compared to the feedforward baseline. However, it was found that the feedback networks led to substantially less false-positive detections.

## References
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

## Feedback networks
The implementation of my two feedback networks can be found here:
- [Configs](configs/pascal_voc/faster_rcnn_r50fbadd_fpn_1x_voc0712.py)
- [Feedback ResNet-50 backbone](mmdet/models/backbones/feedback_resnet.py)
- [McNemar's test for statistical signifacance](tools/analysis_tools/statistical_significance.py)
- [Notebook for postprocessing and analysis of the Robust Detection Benchmark reaults](output/robust-benchmark_utils.ipynb)

## Results Robust Detection Benchmark
| model | mAP |  mPC   |  rPC |
|---|--|---|---|
|FF-Baseline | 0.801 | 0.489 | 60,5 % |
|FB-Mod | 0.806 | 0.468 | 57,5 % |
|FB-Add | 0.803 | 0.457 | 56,3 % |

[mAP] Performance on Clean Data in AP50  
[mPC] Mean Performance under Corruption in AP50  
[rPC] Relative Performance under Corruption in % 

## Reduction of false-positive detections
|Sev 0 | # TP| # FP| Sev 3| # TP| # FP|
|-     |---  |--    |   --|   --|   --|
|FF-Base| 11105| 27675 | | 116107| 346730|
|FB-add| 10712| 13708  | |  102651| 179000|
|FB-mod| 10859 | 16354 | | 106756 | 213403|

Number of true positives (TP) and false positives (FP) per network for corruption severities
0 and 3