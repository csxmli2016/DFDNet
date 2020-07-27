# -- coding: utf-8 --
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import numpy as np
import cv2
import math
from util import util
from scipy.io import loadmat
from PIL import Image
import PIL


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.partpath = opt.partroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.is_real = opt.is_real
        # assert(opt.resize_or_crop == 'resize_and_crop')
        assert(opt.resize_or_crop == 'degradation')   

    def AddNoise(self,img): # noise
        if random.random() > 0.9: #
            return img
        self.sigma = np.random.randint(1, 11)
        img_tensor = torch.from_numpy(np.array(img)).float()
        noise = torch.randn(img_tensor.size()).mul_(self.sigma/1.0)

        noiseimg = torch.clamp(noise+img_tensor,0,255)
        return Image.fromarray(np.uint8(noiseimg.numpy()))

    def AddBlur(self,img): # gaussian blur or motion blur
        if random.random() > 0.9: #
            return img
        img = np.array(img)
        if random.random() > 0.35: ##gaussian blur
            blursize = random.randint(1,17) * 2 + 1 ##3,5,7,9,11,13,15
            blursigma = random.randint(3, 20)
            img = cv2.GaussianBlur(img, (blursize,blursize), blursigma/10)
        else: #motion blur
            M = random.randint(1,32)
            KName = './data/MotionBlurKernel/m_%02d.mat' % M
            k = loadmat(KName)['kernel']
            k = k.astype(np.float32)
            k /= np.sum(k)
            img = cv2.filter2D(img,-1,k)
        return Image.fromarray(img)

    def AddDownSample(self,img): # downsampling
        if random.random() > 0.95: #
            return img
        sampler = random.randint(20, 100)*1.0
        img = img.resize((int(self.opt.fineSize/sampler*10.0), int(self.opt.fineSize/sampler*10.0)), Image.BICUBIC)
        return img

    def AddJPEG(self,img): # JPEG compression
        if random.random() > 0.6: #
            return img
        imQ = random.randint(40, 80)
        img = np.array(img)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),imQ] # (0,100),higher is better,default is 95
        _, encA = cv2.imencode('.jpg',img,encode_param)
        img = cv2.imdecode(encA,1)
        return Image.fromarray(img)

    def AddUpSample(self,img):
        return img.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)

    def __getitem__(self, index): # 

        AB_path = self.AB_paths[index]
        Imgs = Image.open(AB_path).convert('RGB')
        # # 
        A = Imgs.resize((self.opt.fineSize, self.opt.fineSize))
        A = transforms.ColorJitter(0.3, 0.3, 0.3, 0)(A)
        C = A
        A = self.AddUpSample(self.AddJPEG(self.AddNoise(self.AddDownSample(self.AddBlur(A)))))

        tmps = AB_path.split('/')
        ImgName = tmps[-1]
        Part_locations = self.get_part_location(self.partpath, ImgName, 2)
   
        A = transforms.ToTensor()(A) # 
        C = transforms.ToTensor()(C)
        
        ##
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A) # 
        C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C) # 
        return {'A':A, 'C':C, 'A_paths': AB_path,'Part_locations': Part_locations}

    def get_part_location(self, landmarkpath, imgname, downscale=1):
        Landmarks = []
        with open(os.path.join(landmarkpath,imgname+'.txt'),'r') as f:
            for line in f:
                tmp = [np.float(i) for i in line.split(' ') if i != '\n']
                Landmarks.append(tmp)
        Landmarks = np.array(Landmarks)/downscale # 512 * 512
        
        Map_LE = list(np.hstack((range(17,22), range(36,42))))
        Map_RE = list(np.hstack((range(22,27), range(42,48))))
        Map_NO = list(range(29,36))
        Map_MO = list(range(48,68))
        #left eye
        Mean_LE = np.mean(Landmarks[Map_LE],0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        #right eye
        Mean_RE = np.mean(Landmarks[Map_RE],0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        #nose
        Mean_NO = np.mean(Landmarks[Map_NO],0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        #mouth
        Mean_MO = np.mean(Landmarks[Map_MO],0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))

        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
        return Location_LE, Location_RE, Location_NO, Location_MO

    def __len__(self): #
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
