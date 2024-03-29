# Robust Double-Encoder Network for RGB-D Panoptic Segmentation

This repo contains the code of the paper *Robust Double-Encoder Network for RGB-D Panoptic Segmentation*, by [M. Sodano](https://github.com/matteosodano) et al., accepted at the IEEE International Conference on Robotics and Automation (ICRA) 2023. [Link](https://arxiv.org/abs/2210.02834) to the ArXiv version of the paper. 

## Abstract
Perception is crucial for robots that act in real-world environments, as autonomous systems need to see and understand the world around them to act appropriately. Panoptic segmentation provides an interpretation of the scene by computing a pixel-wise semantic label together with instance IDs. In this paper, we address panoptic segmentation using RGB-D data of indoor scenes. We propose a novel encoder-decoder neural network that processes RGB and depth separately through two encoders. The features of the individual encoders are progressively merged at different resolutions, such that the RGB features are enhanced using complementary depth information. We propose a novel merging approach called ResidualExcite, which reweighs each entry of the feature map according to its importance. With our double-encoder architecture, we are robust to missing cues. In particular, the same model can train and infer on RGB-D, RGB-only, and depth-only input data, without the need to train specialized models. We evaluate our method on publicly available datasets and show that our approach achieves superior results compared to other common approaches for panoptic segmentation.

## Results
Quantitative results of our method against baselines:

|Method|Dataset|PQ|mIoU|
|:------:|:-------:|:------:|:-----:|
|RGB Panoptic DeepLab|ScanNet|30.11|43.12|
|RGB-D Panoptic DeepLab|ScanNet|31.43|45.45|
|ESANet with Addition|ScanNet|35.65|51.78|
|ESANet with SE|ScanNet|37.09|54.01|
|Ours with CBAM|ScanNet|39.11|58.11|
|Ours with ResidualExcite|ScanNet|**40.87**|**58.98**|

|Method|Dataset|PQ|mIoU|
|:------:|:-------:|:------:|:-----:|
|RGB Panoptic DeepLab|HyperSim|26.10|40.45|
|RGB-D Panoptic DeepLab|HyperSim|28.56|41.08|
|ESANet with Addition|HyperSim|32.18|50.74|
|ESANet with SE|HyperSim|35.87|54.07|
|Ours with CBAM|HyperSim|37.02|54.21|
|Ours with ResidualExcite|HyperSim|**38.67**|**55.14**|

<br/>

Quantitative results of our method against baselines for Semantic Segmentation only:
|Method|Dataset|mIoU|
|:------:|:-------:|:------:|
|AdapNet++|ScanNet|54.61|
|FuseNet|ScanNet|56.65|
|SSMA|ScanNet|66.13|
|Ours (full)|ScanNet|58.98|
|Ours (semantic-only)|ScanNet|**69.78**|

<br/>

Qualitative results of our method against baselines on the HyperSim dataset, where red circles indicates segmentation errors and green circles well-segmented parts:
![results](pics/hypersim1.jpg)

Check out the [paper](https://arxiv.org/abs/2210.02834) for more quantitative and qualitative results! :grin:

## Code
After installing the requirements specified in the file `requirements.yml` (for example in form of a conda environment), for training your model you only need to `python train.py` and specify your favourite settings in `src/config.yaml`.

Pre-trained models should go in a ad-hoc `trained_models` folder. Some models, such as the pretraining on Imagenet of the ResNet34 model with Non-Bottleneck-1D blocks, are present [here](https://drive.google.com/drive/folders/1goULJjHp5-M7nUGlC52uvWaQxn2j3Za1?usp=sharing). More to come!

## Citation
If you use our framework for any academic work, please cite the original [paper](https://arxiv.org/abs/2210.02834)!

```
@article{sodano2022robust,
  title={Robust Double-Encoder Network for RGB-D Panoptic Segmentation},
  author={Sodano, Matteo and Magistri, Federico and Guadagnino, Tiziano and Behley, Jens and Stachniss, Cyrill},
  journal={arXiv preprint arXiv:2210.02834},
  year={2022}
}
```


## Acknowledgment
All authors are with the [Photogrammetry and Robotics Lab](https://www.ipb.uni-bonn.de/) of the University of Bonn, Germany. 
This work has partially been funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101017008 (Harmony). Project website [here](https://harmony-eu.org/).
