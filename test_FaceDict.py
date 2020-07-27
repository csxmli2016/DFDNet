import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import cv2

from scipy.io import loadmat


def AddNoise(img): # noise

    if random.random() > 0.6: #一定概率不执行
        return img
    sigma = np.random.randint(1, 3)
    img_tensor = torch.from_numpy(np.array(img)).float()
    noise = torch.randn(img_tensor.size()).mul_(sigma/1.0)
    noiseimg = torch.clamp(noise+img_tensor,0,255)
    return Image.fromarray(np.uint8(noiseimg.numpy()))

def AddBlur(img): # gaussian blur or motion blur

    if random.random() > 0.5: #一定概率不执行
        return img
    img = np.array(img)
    if random.random() > 0.5: ##gaussian blur
        blursize = random.randint(1,7) * 2 + 1 ##3,5,7,9,11,13,15
        blursigma = random.randint(3, 17)
        img = cv2.GaussianBlur(img, (blursize,blursize), blursigma/10)
    else: #motion blur
        M = random.randint(1,32)
        KName = './data/MotionBlurKernel/m_%02d.mat' % M
        k = loadmat(KName)['kernel']
        k = k.astype(np.float32)
        k /= np.sum(k)
        img = cv2.filter2D(img,-1,k)
    return Image.fromarray(img)

def AddBlurFix(img): # gaussian blur or motion blur

    img = np.array(img)
    if random.random() > 0: ##gaussian blur
        blursize = random.randint(7,7) * 2 + 1 ##3,5,7,9,11,13,15
        blursigma = random.randint(30, 30)
        img = cv2.GaussianBlur(img, (blursize,blursize), blursigma/10)
 
    return Image.fromarray(img)

def AddDownSample(img): # downsampling
    # if random.random() > 0.95: #一定概率不执行
    #     return img
    # sampler = random.randint(40, 80)*1.0
    sampler = 100
    img = img.resize((int(256/sampler*10.0), int(256/sampler*10.0)), Image.BICUBIC)
    return img

def AddJPEG(img): # JPEG compression
    if random.random() > 0.5: #一定概率不执行
        return img
    imQ = random.randint(60, 90)
    img = np.array(img)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),imQ] # (0,100),higher is better,default is 95
    _, encA = cv2.imencode('.jpg',img,encode_param)
    img = cv2.imdecode(encA,1)
    return Image.fromarray(img)

def AddUpSample(img):
    return img.resize((512, 512), Image.BICUBIC)

def GaussianBlur(img): # gaussian blur or motion blur
    img = np.array(img)
    
    blursize = random.randint(7,7) * 2 + 1 ##3,5,7,9,11,13,15
    blursigma = random.randint(20, 20)
    img = cv2.GaussianBlur(img, (blursize,blursize), blursigma/10)
    return Image.fromarray(img)

def get_part_location(partpath, imgname):
    Landmarks = []
    if not os.path.exists(os.path.join(partpath,imgname+'.txt')):
        print(os.path.join(partpath,imgname+'.txt'))
        print('no landmark file')
        return 0

    with open(os.path.join(partpath,imgname+'.txt'),'r') as f:
        for line in f:
            tmp = [np.float(i) for i in line.split(' ') if i != '\n']
            Landmarks.append(tmp)
    
    Landmarks = np.array(Landmarks) # 256 * 256 VggFac的landmark要乘以2
    if Landmarks.max() < 256:  ###这里需要的是512*512的点
        Landmarks = Landmarks * 2
    # Landmarks = Landmarks * 0.1

    Map_LE = list(np.hstack((range(17,22), range(36,42))))
    Map_RE = list(np.hstack((range(22,27), range(42,48))))
    Map_NO = list(range(29,36))
    Map_MO = list(range(48,68))
    #left eye
    # print(Landmarks)
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

    return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)

def obtain_inputs(img_path, Landmark_path, img_name, Type):
    A_paths = os.path.join(img_path,img_name)
    Imgs = Image.open(A_paths).convert('RGB')

    Part_locations = get_part_location(Landmark_path, img_name)
    if Part_locations == 0:
        print('wrong part_location')
        return 0
    width, height = Imgs.size
    L = min(width, height)
    #################################################
    A= Imgs
    C = A
    #################################################
    # A = Imgs.crop((0, 0, L, L))
    # if width == height:
    #     C = A
    # else:
    #     C = Imgs.crop((0,L,L,2*L))
    #################################################
    if Type == 'random':
        A = AddUpSample(AddJPEG(AddNoise(AddDownSample(AddBlur(A)))))
        # A = AddUpSample(AddBlurFix(A))
        # A = AddUpSample(AddJPEG(AddNoise(AddDownSample(AddBlur(A)))))
        # A = AddUpSample(AddNoise((AddDownSample(AddBlur(A)))))
        # A = AddUpSample(AddJPEG(AddDownSample(AddBlur(A))))
    
    A = AddUpSample(A)
        # A = AddUpSample(AddDownSample(A))

    # A_Blur = GaussianBlur(A) ####归一化模糊方式
    A = transforms.ToTensor()(A) #
    # A_Blur = transforms.ToTensor()(A_Blur) #
    C = transforms.ToTensor()(C)

    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A) # 正则化,(image-mean)/std
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C) # 范围为-1到1
    # A_Blur = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A_Blur)

    return {'A':A.unsqueeze(0), 'C':C.unsqueeze(0), 'A_paths': A_paths,'Part_locations': Part_locations}
    


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.which_epoch = 'latest' #SFFConcat_net_G, #SFF2layer_net_G #SFF1layer_net_G 
    #Ours V1_net_G(nogan) V2_net_G(ssim) latest #latestNoAttention

    ####################################################
    ##Test Param
    #####################################################
    IsReal = 0
    opt.gpu_ids = [0]
    TestImgPath = './TestData/crop2' #%
    LandmarkPath = './TestData/crop2_landmark'
    opt.results_dir = './Results/crop2'


    #####################################

    model = create_model(opt)
    model.setup(opt)
    # create website
    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (TestFile, opt.which_epoch))
    web_dir = os.path.join(opt.results_dir, opt.name)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test

    ImgNames = os.listdir(TestImgPath)
    ImgNames.sort()
    total = 0

    for i, ImgName in enumerate(ImgNames):
        print((i,ImgName))

        if total >= 50:
            break

        data = obtain_inputs(TestImgPath, LandmarkPath, ImgName, 'real')
        if data == 0:
            continue
        total = total + 1

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
