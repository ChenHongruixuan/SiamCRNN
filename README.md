# SiamCRNN
Code for TGRS 2020 [Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network](https://ieeexplore.ieee.org/document/8937755).

<img src="./Fig/SiamCRNN.bmp" width="100%" height="100%">

## Abstract
> With the rapid development of Earth observation technology, very-high-resolution (VHR) images from various satellite sensors are more available, which greatly enrich the data source of change detection (CD). Multisource multitemporal images can provide abundant information on observed landscapes with various physical and material views, and it is exigent to develop efficient techniques to utilize these multisource data for CD. In this article, we propose a novel and general deep siamese convolutional multiple-layers recurrent neural network (RNN) (SiamCRNN) for CD in multitemporal VHR images. Superior to most VHR image CD methods, SiamCRNN can be used for both homogeneous and heterogeneous images. Integrating the merits of both convolutional neural network (CNN) and RNN, Siam-CRNN consists of three subnetworks: deep siamese convolutional neural network (DSCNN), multiple-layers RNN (MRNN), and fully connected (FC) layers. The DSCNN has a flexible structure for multisource image and is able to extract spatial–spectral features from homogeneous or heterogeneous VHR image patches. The MRNN stacked by long-short term memory (LSTM) units is responsible for mapping the spatial–spectral features extracted by DSCNN into a new latent feature space and mining the change information between them. In addition, FC, the last part of SiamCRNN, is adopted to predict change probability. The experimental results in two homogeneous data sets and one challenging heterogeneous VHR images data set demonstrate that the promising performances of the proposed network outperform several state-of-the-art approaches.

## Requirements
```
tensorflow_gpu==1.9.0
opencv==3.4.0
numpy==1.14.0
```

## Dataset
The datasets WH and HY used in our paper has been open-sourced! You can download them [here](http://sigma.whu.edu.cn/resource.php).   

## Citation
Please considering to cite our paper if you use this code in your research.
```
@article{Chen2020,
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
