#!/usr/bin/python #encoding:utf-8
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cv2
import os
import face_alignment
from skimage import io, transform

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,device='cuda:0', flip_input=False)


Nums = 0
FilePath = '../TestData/RealVgg/Imgs'
SavePath = '../TestData/RealVgg/Imgs/Landmarks'
if not os.path.exists(SavePath):
    os.mkdir(SavePath)

ImgNames = os.listdir(FilePath)
ImgNames.sort()

for i,name in enumerate(ImgNames):
    print((i,name))

    imgs = io.imread(os.path.join(FilePath,name))

    imgO = imgs
    try:
        PredsAll = fa.get_landmarks(imgO)
    except:
        print('#########No face')
        continue
    if PredsAll is None:
        print('#########No face2')
        continue
    if len(PredsAll)!=1:
        print('#########too many face')
        continue
    preds = PredsAll[-1]
    AddLength = np.sqrt(np.sum(np.power(preds[27][0:2]-preds[33][0:2],2)))
    SaveName = name+'.txt'

    np.savetxt(os.path.join(SavePath,SaveName),preds[:,0:2],fmt='%.3f')
