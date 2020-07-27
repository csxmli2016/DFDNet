# -- coding: utf-8 --
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torchvision

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, norm=1, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    if norm == 1: #for clamp -1 to 1
        image_numpy = image_tensor[0].cpu().float().clamp_(-1,1).numpy()
    elif norm == 2: # for norm through max-min
        image_ = image_tensor[0].cpu().float()
        max_ = torch.max(image_)
        min_ = torch.min(image_)
        image_numpy = (image_ - min_)/(max_-min_)*2-1
        image_numpy = image_numpy.numpy() 
    else:
        pass
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # print(image_numpy.shape)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # print(image_numpy.shape)
    return image_numpy.astype(imtype)
def tensor2im3Channels(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    
    image_numpy = image_tensor.cpu().float().clamp_(-1,1).numpy()

    # print(image_numpy.shape)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # print(image_numpy.shape)
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)




def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_current_losses(epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        # with open('', "a") as log_file:
        #     log_file.write('%s\n' % message)

def display_current_results(writer,visuals,losses,step,save_result):
    for label, images in visuals.items():
        if 'Mask' in label:#  or 'Scale' in label:
            grid = torchvision.utils.make_grid(images,normalize=False, scale_each=True)
            # pass
        else:
            pass
        grid = torchvision.utils.make_grid(images,normalize=True, scale_each=True)
        writer.add_image(label,grid,step)
    for k,v in losses.items():
        writer.add_scalar(k,v,step)

def VisualFeature(input_feature, imtype=np.uint8):
    if isinstance(input_feature, torch.Tensor):
        image_tensor = input_feature.data
    else:
        return input_feature
    
    image_ = image_tensor.cpu().float()

    if image_.size(1) == 3:
        image_ = image_.permute(1,2,0)

    # assert(image_.size(1) == 1)


    
    #####norm 0 to 1
    max_ = torch.max(image_)
    min_ = torch.min(image_)
    image_numpy = (image_ - min_)/(max_-min_)*2-1
    image_numpy = image_numpy.numpy()
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    #####no norm
    # print((max_,min_))
    # image_numpy = image_.numpy()
    # image_numpy = image_numpy*255.0


    # print('wwwwwwwwwwwwww')
    # print(max_)
    # print(min_)
    # print(image_numpy.shape)
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
