# DFDNet
Overview of our proposed method. It mainly contains two parts: (a) the online generation of multi-scale components dictionaries from large amounts of high-quality images with diverse poses and expressions. K-means is adopted to generate K clusters for each component (i.e., left/right eyes, nose and mouth) on different feature scales. (b) The restoration process and dictionary feature transfer block that are utilized to provide the reference details in a progressive manner. Here, DFT-i block takes the Scale-i components dictionaries for reference in the same feature level.

<img src="./Imgs/pipeline_a.png">
<img src="./Imgs/pipeline_b.png">

## All the codes and models will be public available after camera-ready.


# Citation

```
@InProceedings{Li_2020_ECCV,
author = {Li, Xiaoming and Chen, Chaofeng and Zhou, Shangchen and Lin, Xianhui and Zuo, Wangmeng and Zhang, Lei},
title = {Blind Face Restoration via Deep Multi-scale Component Dictionaries},
booktitle = {ECCV},
year = {2020}
}
```
