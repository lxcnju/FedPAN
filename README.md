# FedPAN
The source code of our works on federated learning:
* CVPR 2022 paper: Federated Learning with Position-Aware Neurons.


# Content
* Personal Homepage
* Basic Introduction
* Running Tips
* Citation

## Personal Homepage
  * [Homepage](https://www.lamda.nju.edu.cn/lixc/)

## Basic Introduction
  * More could be found in our FL repository [FedRepo](https://github.com/lxcnju/FedRepo)

## Environment Dependencies
The code files are written in Python, and the utilized deep learning tool is PyTorch.
  * `python`: 3.7.3
  * `numpy`: 1.21.5
  * `torch`: 1.9.0
  * `torchvision`: 0.10.0
  * `pillow`: 8.3.1

## Datasets
We provide several datasets including (downloading link code be found in my [Homepage](https://www.lamda.nju.edu.cn/lixc/)):
  * CIFAR-10
  * CIFAR-100

## Running Tips
  * `python train_fedpan.py`: notice that FedPAN differs from FedAvg only in the utilized networks, where the former uses networks with PANs (Position-Aware Neurons).
  * The hyper-parameters of PANs: pe_way (default is "sin"), pe_t (default is 1.0, T in the paper), pe_op ("add" or "mul", two types of PANs in the paper), pe_alpha (A in the paper), pe_ratio (a constant of 1.0). pe_op and pe_alpha matters in FedPAN.

FL algorithms and hyper-parameters could be set in these files.


## Citation
  * Xin-Chun Li, Yi-Chu Xu, Shaoming Song, Bingshuai Li, Yinchuan Li, Yunfeng Shao, De-Chuan Zhan. Federated Learning with Position-Aware Neurons. In: Proceedings of the 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR'2022), online conference, New Orleans, Louisiana, 2022.
  * \[[BibTex](https://dblp.org/pid/246/2947.html)\]
