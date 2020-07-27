# DFDNet
Overview of our proposed method. It mainly contains two parts: (a) the off-line generation of multi-scale component dictionaries from large amounts of high-quality images which have diverse poses and expressions. K-means is adopted to generate K clusters for each component (i.e., left/right eyes, nose and mouth) on different feature scales. (b) The restoration process and dictionary feature transfer (DFT) block that are utilized to provide the reference details in a progressive manner. Here, DFT-i block takes the Scale-i component dictionaries for reference in the same feature level.
    
    

<img src="./Imgs/pipeline_a.png">
<p align="center">(a) Offline generation of multi-scale component dictionaries.</p>
<img src="./Imgs/pipeline_b.png">
<p align="center">(b) Architecture of our DFDNet for dictionary feature transfer.</p>


# Models
Download the pre-trained model with the following url and put it into ./checkpoints/.
- [BaiduNetDisk](https://pan.baidu.com/s/1AXq5Hpa0dCSCu1fuj5CkOA) (c2x2)
- [GoogleDrive](https://drive.google.com/file/d/1UCo7YEbLLa1_87b0AoWmzhTGyrw-26nb/view?usp=sharing)

# Component Dictionaries
Download the dictionaries with the following url and put it into ./.
- [BaiduNetDisk](https://pan.baidu.com/s/1p-u6wpLU_ayAm2Lt4D-MLg) (3y2r)
- [GoogleDrive](https://drive.google.com/drive/folders/1iwQjHx23O1HVWJ0rtwos8OVZ3mIeCe8r?usp=sharing)

#Testing
```bash
python test_FaceDict.py
```

# Citation

```
@InProceedings{Li_2020_ECCV,
author = {Li, Xiaoming and Chen, Chaofeng and Zhou, Shangchen and Lin, Xianhui and Zuo, Wangmeng and Zhang, Lei},
title = {Blind Face Restoration via Deep Multi-scale Component Dictionaries},
booktitle = {ECCV},
year = {2020}
}
```
