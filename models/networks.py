import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn import Parameter as P
from util import util
from torchvision import models
import scipy.io as sio
import numpy as np
import scipy.ndimage
import torch.nn.utils.spectral_norm as SpectralNorm

from torch.autograd import Function
from math import sqrt
import random
import os
import math

from sync_batchnorm import convert_model
####

###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], init_flag=True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = convert_model(net)
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    if init_flag:

        init_weights(net, init_type, gain=init_gain)

    return net


# compute adaptive instance norm
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    C, _ = size[:2]
    feat_var = feat.contiguous().view(C, -1).var(dim=1) + eps
    feat_std = feat_var.sqrt().view(C, 1, 1)
    feat_mean = feat.contiguous().view(C, -1).mean(dim=1).view(C, 1, 1)

    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):  # content_feat is degraded feature, style is ref feature
    assert (content_feat.size()[:1] == style_feat.size()[:1])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std_4D(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization_4D(content_feat, style_feat): # content_feat is ref feature, style is degradate feature
    # assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_4D(style_feat)

    content_mean, content_std = calc_mean_std_4D(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def define_G(which_model_netG, gpu_ids=[]):
    if which_model_netG == 'UNetDictFace':
        netG = UNetDictFace(64)
        init_flag = False
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, 'normal', 0.02, gpu_ids, init_flag)


##############################################################################
# Classes
############################################################################################################################################


def convU(in_channels, out_channels,conv_layer, norm_layer, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        SpectralNorm(conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)),
#         conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
#         nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
        SpectralNorm(conv_layer(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)),
    )
class MSDilateBlock(nn.Module):
    def __init__(self, in_channels,conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, kernel_size=3, dilation=[1,1,1,1], bias=True):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  SpectralNorm(conv_layer(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias))
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out

##############################UNetFace#########################
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)

    def forward(self, input, style):
        style_mean, style_std = calc_mean_std_4D(style)
        out = self.norm(input)
        size = input.size()
        out = style_std.expand(size) * out + style_mean.expand(size)
        return out

class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )
        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)
    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)
    def forward(self, input):
        return self.conv(input)

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))
    def forward(self, image, noise):
        return image + self.weight * noise

class StyledUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,upsample=False):
        super().__init__()
        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                Blur(out_channel),
                # EqualConv2d(in_channel, out_channel, kernel_size, padding=padding),
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
            )
        else:
            self.conv1 = nn.Sequential(
                Blur(in_channel),
                # EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
            )
        self.convup = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # EqualConv2d(out_channel, out_channel, kernel_size, padding=padding),
                SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
                # Blur(out_channel),
            )
        # self.noise1 = equal_lr(NoiseInjection(out_channel))
        # self.adain1 = AdaptiveInstanceNorm(out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)

        # self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        # self.noise2 = equal_lr(NoiseInjection(out_channel))
        # self.adain2 = AdaptiveInstanceNorm(out_channel)
        # self.lrelu2 = nn.LeakyReLU(0.2)

        self.ScaleModel1 = nn.Sequential(
            # Blur(in_channel),
            SpectralNorm(nn.Conv2d(in_channel,out_channel,3, 1, 1)),
            # nn.Conv2d(in_channel,out_channel,3, 1, 1),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
            # nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )
        self.ShiftModel1 = nn.Sequential(
            # Blur(in_channel),
            SpectralNorm(nn.Conv2d(in_channel,out_channel,3, 1, 1)),
            # nn.Conv2d(in_channel,out_channel,3, 1, 1),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
            nn.Sigmoid(),
            # nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )
       
    def forward(self, input, style):
        out = self.conv1(input)
#         out = self.noise1(out, noise)
        out = self.lrelu1(out)

        Shift1 = self.ShiftModel1(style)
        Scale1 = self.ScaleModel1(style)
        out = out * Scale1 + Shift1
        # out = self.adain1(out, style)
        outup = self.convup(out)

        return outup

##############################################################################
##Face Dictionary
##############################################################################
class VGGFeat(torch.nn.Module):
    """
    Input: (B, C, H, W), RGB, [-1, 1]
    """
    def __init__(self, weight_path='./weights/vgg19.pth'):
        super().__init__()
        self.model = models.vgg19(pretrained=False)
        self.build_vgg_layers()
        
        self.model.load_state_dict(torch.load(weight_path))

        self.register_parameter("RGB_mean", nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)))
        self.register_parameter("RGB_std", nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)))
        
        # self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def build_vgg_layers(self):
        vgg_pretrained_features = self.model.features
        self.features = []
        # feature_layers = [0, 3, 8, 17, 26, 35]
        feature_layers = [0, 8, 17, 26, 35]
        for i in range(len(feature_layers)-1): 
            module_layers = torch.nn.Sequential() 
            for j in range(feature_layers[i], feature_layers[i+1]):
                module_layers.add_module(str(j), vgg_pretrained_features[j])
            self.features.append(module_layers)
        self.features = torch.nn.ModuleList(self.features)

    def preprocess(self, x):
        x = (x + 1) / 2
        x = (x - self.RGB_mean) / self.RGB_std
        if x.shape[3] < 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        features = []
        for m in self.features:
            # print(m)
            x = m(x)
            features.append(x)
        return features 

def compute_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x
def ToRGB(in_channel):
    return nn.Sequential(
        SpectralNorm(nn.Conv2d(in_channel,in_channel,3, 1, 1)),
        nn.LeakyReLU(0.2),
        SpectralNorm(nn.Conv2d(in_channel,3,3, 1, 1))
    )

def AttentionBlock(in_channel):
    return nn.Sequential(
        SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)),
        nn.LeakyReLU(0.2),
        SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1))
    )

class UNetDictFace(nn.Module):
    def __init__(self, ngf=64, dictionary_path='./DictionaryCenter512'):
        super().__init__()
        
        self.part_sizes = np.array([80,80,50,110]) # size for 512
        self.feature_sizes = np.array([256,128,64,32])
        self.channel_sizes = np.array([128,256,512,512])
        Parts = ['left_eye','right_eye','nose','mouth']
        self.Dict_256 = {}
        self.Dict_128 = {}
        self.Dict_64 = {}
        self.Dict_32 = {}
        for j,i in enumerate(Parts):
            f_256 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_256_center.npy'.format(i)), allow_pickle=True))

            f_256_reshape = f_256.reshape(f_256.size(0),self.channel_sizes[0],self.part_sizes[j]//2,self.part_sizes[j]//2)
            max_256 = torch.max(torch.sqrt(compute_sum(torch.pow(f_256_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_256[i] = f_256_reshape #/ max_256
            
            f_128 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_128_center.npy'.format(i)), allow_pickle=True))

            f_128_reshape = f_128.reshape(f_128.size(0),self.channel_sizes[1],self.part_sizes[j]//4,self.part_sizes[j]//4)
            max_128 = torch.max(torch.sqrt(compute_sum(torch.pow(f_128_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_128[i] = f_128_reshape #/ max_128

            f_64 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_64_center.npy'.format(i)), allow_pickle=True))

            f_64_reshape = f_64.reshape(f_64.size(0),self.channel_sizes[2],self.part_sizes[j]//8,self.part_sizes[j]//8)
            max_64 = torch.max(torch.sqrt(compute_sum(torch.pow(f_64_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_64[i] = f_64_reshape #/ max_64

            f_32 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_32_center.npy'.format(i)), allow_pickle=True))

            f_32_reshape = f_32.reshape(f_32.size(0),self.channel_sizes[3],self.part_sizes[j]//16,self.part_sizes[j]//16)
            max_32 = torch.max(torch.sqrt(compute_sum(torch.pow(f_32_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_32[i] = f_32_reshape #/ max_32

        self.le_256 = AttentionBlock(128)
        self.le_128 = AttentionBlock(256)
        self.le_64 = AttentionBlock(512)
        self.le_32 = AttentionBlock(512)

        self.re_256 = AttentionBlock(128)
        self.re_128 = AttentionBlock(256)
        self.re_64 = AttentionBlock(512)
        self.re_32 = AttentionBlock(512)

        self.no_256 = AttentionBlock(128)
        self.no_128 = AttentionBlock(256)
        self.no_64 = AttentionBlock(512)
        self.no_32 = AttentionBlock(512)

        self.mo_256 = AttentionBlock(128)
        self.mo_128 = AttentionBlock(256)
        self.mo_64 = AttentionBlock(512)
        self.mo_32 = AttentionBlock(512)

        #norm
        self.VggExtract = VGGFeat()
        
        ######################
        self.MSDilate = MSDilateBlock(ngf*8, dilation = [4,3,2,1])  #

        self.up0 = StyledUpBlock(ngf*8,ngf*8)
        self.up1 = StyledUpBlock(ngf*8, ngf*4) #
        self.up2 = StyledUpBlock(ngf*4, ngf*2) #
        self.up3 = StyledUpBlock(ngf*2, ngf) #
        self.up4 = nn.Sequential( # 128
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1)),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            UpResBlock(ngf),
            UpResBlock(ngf),
            # SpectralNorm(nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.to_rgb0 = ToRGB(ngf*8)
        self.to_rgb1 = ToRGB(ngf*4)
        self.to_rgb2 = ToRGB(ngf*2)
        self.to_rgb3 = ToRGB(ngf*1)

        # for param in self.BlurInputConv.parameters():
        #     param.requires_grad = False
    
    def forward(self,input, part_locations):

        VggFeatures = self.VggExtract(input)
        # for b in range(input.size(0)):
        b = 0
        UpdateVggFeatures = []
        for i, f_size in enumerate(self.feature_sizes):
            cur_feature = VggFeatures[i]
            update_feature = cur_feature.clone() #* 0
            cur_part_sizes = self.part_sizes // (512/f_size)

            dicts_feature = getattr(self, 'Dict_'+str(f_size))
            LE_Dict_feature = dicts_feature['left_eye'].to(input)
            RE_Dict_feature = dicts_feature['right_eye'].to(input)
            NO_Dict_feature = dicts_feature['nose'].to(input)
            MO_Dict_feature = dicts_feature['mouth'].to(input)

            le_location = (part_locations[0][b] // (512/f_size)).int()
            re_location = (part_locations[1][b] // (512/f_size)).int()
            no_location = (part_locations[2][b] // (512/f_size)).int()
            mo_location = (part_locations[3][b] // (512/f_size)).int()

            LE_feature = cur_feature[:,:,le_location[1]:le_location[3],le_location[0]:le_location[2]].clone()
            RE_feature = cur_feature[:,:,re_location[1]:re_location[3],re_location[0]:re_location[2]].clone()
            NO_feature = cur_feature[:,:,no_location[1]:no_location[3],no_location[0]:no_location[2]].clone()
            MO_feature = cur_feature[:,:,mo_location[1]:mo_location[3],mo_location[0]:mo_location[2]].clone()
            
            #resize
            LE_feature_resize = F.interpolate(LE_feature,(LE_Dict_feature.size(2),LE_Dict_feature.size(3)),mode='bilinear',align_corners=False)
            RE_feature_resize = F.interpolate(RE_feature,(RE_Dict_feature.size(2),RE_Dict_feature.size(3)),mode='bilinear',align_corners=False)
            NO_feature_resize = F.interpolate(NO_feature,(NO_Dict_feature.size(2),NO_Dict_feature.size(3)),mode='bilinear',align_corners=False)
            MO_feature_resize = F.interpolate(MO_feature,(MO_Dict_feature.size(2),MO_Dict_feature.size(3)),mode='bilinear',align_corners=False)

            LE_Dict_feature_norm = adaptive_instance_normalization_4D(LE_Dict_feature, LE_feature_resize)
            RE_Dict_feature_norm = adaptive_instance_normalization_4D(RE_Dict_feature, RE_feature_resize)
            NO_Dict_feature_norm = adaptive_instance_normalization_4D(NO_Dict_feature, NO_feature_resize)
            MO_Dict_feature_norm = adaptive_instance_normalization_4D(MO_Dict_feature, MO_feature_resize)
            
            LE_score = F.conv2d(LE_feature_resize, LE_Dict_feature_norm)

            LE_score = F.softmax(LE_score.view(-1),dim=0)
            LE_index = torch.argmax(LE_score)
            LE_Swap_feature = F.interpolate(LE_Dict_feature_norm[LE_index:LE_index+1], (LE_feature.size(2), LE_feature.size(3)))

            LE_Attention = getattr(self, 'le_'+str(f_size))(LE_Swap_feature-LE_feature)
            LE_Att_feature = LE_Attention * LE_Swap_feature
            

            RE_score = F.conv2d(RE_feature_resize, RE_Dict_feature_norm)
            RE_score = F.softmax(RE_score.view(-1),dim=0)
            RE_index = torch.argmax(RE_score)
            RE_Swap_feature = F.interpolate(RE_Dict_feature_norm[RE_index:RE_index+1], (RE_feature.size(2), RE_feature.size(3)))
            
            RE_Attention = getattr(self, 're_'+str(f_size))(RE_Swap_feature-RE_feature)
            RE_Att_feature = RE_Attention * RE_Swap_feature

            NO_score = F.conv2d(NO_feature_resize, NO_Dict_feature_norm)
            NO_score = F.softmax(NO_score.view(-1),dim=0)
            NO_index = torch.argmax(NO_score)
            NO_Swap_feature = F.interpolate(NO_Dict_feature_norm[NO_index:NO_index+1], (NO_feature.size(2), NO_feature.size(3)))
            
            NO_Attention = getattr(self, 'no_'+str(f_size))(NO_Swap_feature-NO_feature)
            NO_Att_feature = NO_Attention * NO_Swap_feature

            
            MO_score = F.conv2d(MO_feature_resize, MO_Dict_feature_norm)
            MO_score = F.softmax(MO_score.view(-1),dim=0)
            MO_index = torch.argmax(MO_score)
            MO_Swap_feature = F.interpolate(MO_Dict_feature_norm[MO_index:MO_index+1], (MO_feature.size(2), MO_feature.size(3)))
            
            MO_Attention = getattr(self, 'mo_'+str(f_size))(MO_Swap_feature-MO_feature)
            MO_Att_feature = MO_Attention * MO_Swap_feature

            update_feature[:,:,le_location[1]:le_location[3],le_location[0]:le_location[2]] = LE_Att_feature + LE_feature
            update_feature[:,:,re_location[1]:re_location[3],re_location[0]:re_location[2]] = RE_Att_feature + RE_feature
            update_feature[:,:,no_location[1]:no_location[3],no_location[0]:no_location[2]] = NO_Att_feature + NO_feature
            update_feature[:,:,mo_location[1]:mo_location[3],mo_location[0]:mo_location[2]] = MO_Att_feature + MO_feature

            UpdateVggFeatures.append(update_feature) 
        
        fea_vgg = self.MSDilate(VggFeatures[3])
        #new version
        fea_up0 = self.up0(fea_vgg, UpdateVggFeatures[3])
        # out1 = F.interpolate(fea_up0,(512,512))
        # out1 = self.to_rgb0(out1)

        fea_up1 = self.up1( fea_up0, UpdateVggFeatures[2]) #
        # out2 = F.interpolate(fea_up1,(512,512))
        # out2 = self.to_rgb1(out2)

        fea_up2 = self.up2(fea_up1, UpdateVggFeatures[1]) #
        # out3 = F.interpolate(fea_up2,(512,512))
        # out3 = self.to_rgb2(out3)

        fea_up3 = self.up3(fea_up2, UpdateVggFeatures[0]) #
        # out4 = F.interpolate(fea_up3,(512,512))
        # out4 = self.to_rgb3(out4)

        output = self.up4(fea_up3) #
        
    
        return output  #+ out4 + out3 + out2 + out1
        #0 128 * 256 * 256
        #1 256 * 128 * 128
        #2 512 * 64 * 64
        #3 512 * 32 * 32


class UpResBlock(nn.Module):
    def __init__(self, dim, conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d):
        super(UpResBlock, self).__init__()
        self.Model = nn.Sequential(
            # SpectralNorm(conv_layer(dim, dim, 3, 1, 1)),
            conv_layer(dim, dim, 3, 1, 1),
            # norm_layer(dim),
            nn.LeakyReLU(0.2,True),
            # SpectralNorm(conv_layer(dim, dim, 3, 1, 1)),
            conv_layer(dim, dim, 3, 1, 1),
        )
    def forward(self, x):
        out = x + self.Model(x)
        return out

class VggClassNet(nn.Module):
    def __init__(self, select_layer = ['0','5','10','19']):
        super(VggClassNet, self).__init__()
        self.select = select_layer
        self.vgg = models.vgg19(pretrained=True).features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


if __name__ == '__main__':
    print('this is network')


