<h1 align="center">SiamCRNN in Fully Convolutional Version</h1>

<h3 align="center"> <a href="https://chrx97.com/">Hongruixuan Chen</a>, <a href="https://scholar.google.com/citations?user=DbTt_CcAAAAJ&hl=zh-CN">Chen Wu</a>, <a href="https://scholar.google.com/citations?user=Shy1gnMAAAAJ&hl=zh-CN">Bo Du</a>,
<a href="https://scholar.google.com/citations?user=vzj2hcYAAAAJ&hl=en">Liangpei Zhang</a>, and <a href="https://scholar.google.com/citations?hl=en&user=AvOyKAUAAAAJ">Le Wang</a></h3>

This is an implementation of fully convolutional version of **SiamCRNN** framework in our IEEE TGRS 2020 paper: [Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network](https://ieeexplore.ieee.org/document/8937755).

## Introduction
We have improved the SiamCRNN from our [original paper](https://ieeexplore.ieee.org/document/8937755) so that SiamCRNN can be used for large-scale change detection tasks. The entire architecture is fully convolutional, where the encoder is an arbitrary fully convolutional deep network (we use ResNet here) and the decoder is a multilayer ConvLSTM+FPN.

## Get started
### Requirements
```
python==3.8.18
pytorch==1.21.1
torchvision==0.13.1
imageio==2.22.4
numpy==1.14.0
tqdm==4.64.1
```

### Dataset
The fully convolutional version of SiamCRNN can be trained and tested on arbitrary large-scale change detection benchmark datasets, such as [SYSU](https://github.com/liumency/SYSU-CD), [LEVIR-CD](https://chenhao.in/LEVIR/), etc. We provide here the example on the [OSCD dataset](https://rcdaudt.github.io/oscd/).

### Training
```
python train_siamcrnn.py
```

### Testing
```
python test_siamcrnn.py
```

### Detection results
|               |         |                      |         |                           |         |                      |                
|:-------------:|:---------------:|:--------------------:|:---------------:|:--------------------:|:--------------------:|:--------------------:|
|     Method    |       Rec        |          Pre          |       OA        |          F1          |          IoU          |          KC          |
|     FC-EF      |     0.4612		 | 0.4967      |    0.9480		|    0.4783	   | 0.3143	  |  0.4510       |               
|     FC-Siam-Conc      |   0.5362 	 |  	0.4760	   |     0.9455	|   0.5043	   | 0.3372		  |   0.4757	         |               
|     FC-Siam-Diff      |    0.5385 | 	0.4391      |    	0.9406	|    	0.4838		   |  0.3191	  |   0.4526 |               
|     [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images)     |    0.4538  | 0.6732    | 0.9604	|      0.5421	   |  0.3718	  | 0.5222   |               
|     [ChangeFormer](https://github.com/wgcban/ChangeFormer)     |    0.5351		 | 0.5566     |     0.9540		|     0.5457	   |  0.3752	  |  0.5215	      |               
|     **SiamCRNN (ResNet18)**      |     **0.5160**	  |  **0.5758**      |      **0.9553**	|      **0.5442**	   |  **0.3739**	  |   **0.5208**         |
					
## Citation
If this code or dataset contributes to your research, please consider citing our paper. We appreciate your support!🙂
```
@article{Chen2020Change,
author = {Chen, Hongruixuan and Wu, Chen and Du, Bo and Zhang, Liangpei and Wang, Le},
issn = {0196-2892},
journal = {IEEE Transactions on Geoscience and Remote Sensing},
number = {4},
pages = {2848--2864},
title = {{Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network}},
volume = {58},
year = {2020}
}
```

## Q & A
**For any questions, please [contact us.](mailto:Qschrx@gmail.com)**
