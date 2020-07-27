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
####
#tes

from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d
from torchvision import models
from sync_batchnorm import convert_model
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
    print(which_model_netG)
    if which_model_netG == 'UNetDictFace':
        netG = UNetDictFace(64)
        init_flag = False
    elif which_model_netG == 'StyleFace':
        netG = StyleFace(64)
        init_flag = True
    elif which_model_netG == 'UNetDictFaceV3':
        netG = UNetDictFaceV3(64)
        init_flag = False
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    return init_net(netG, 'normal', 0.02, gpu_ids, init_flag)


def define_D(which_model_netD, resolution=256, init_type='orthogonal', init_gain=0.02, gpu_ids=[], Scales = [512,256,128]):
    init_flag = True
    if which_model_netD == 'Discriminator':
        netD = Discriminator(resolution=resolution)
    elif which_model_netD == 'Discriminator_LE':
        netD = Discriminator_LE(resolution=resolution)
    elif which_model_netD == 'Discriminator_RE':
        netD = Discriminator_RE(resolution=resolution)
    elif which_model_netD == 'Discriminator_NO':
        netD = Discriminator_NO(resolution=resolution)
    elif which_model_netD == 'Discriminator_MO':
        netD = Discriminator_MO(resolution=resolution)
    elif which_model_netD == 'SND':
        netD = SNDiscriminator()
        init_flag = False
    elif which_model_netD == 'MSSND':
        netD = MultiScaleSNDiscriminator(Scales=Scales)
    elif which_model_netD == 'SAGAN':
        netD = SAGAN()
    elif which_model_netD == 'StyleDiscriminator':
        netD = StyleDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)

    return init_net(netD, init_type, init_gain, gpu_ids, init_flag)

##############################################################################
# Classes
##############################################################################
class PureUpsampling(nn.Module):
    def __init__(self, scale=2, mode='bilinear'):
        super(PureUpsampling, self).__init__()
        assert isinstance(scale, int)
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        h, w = x.size(2) * self.scale, x.size(3) * self.scale
        if self.mode == 'nearest':
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode)
        else:
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode, align_corners=True)
        return xout


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention


class SAGAN(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size=64, conv_dim=64):
        super(SAGAN, self).__init__()

        curr_dim = conv_dim
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer6 = []
        layer7 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        self.l1 = nn.Sequential(*layer1)
        

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        self.l2 = nn.Sequential(*layer2)

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim * 2, curr_dim * 4, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        self.l3 = nn.Sequential(*layer3)

        
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim * 4, curr_dim * 8, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)

        layer5.append(SpectralNorm(nn.Conv2d(curr_dim * 8, curr_dim * 8, 4, 2, 1)))
        layer5.append(nn.LeakyReLU(0.1))
        self.l5 = nn.Sequential(*layer5)

        layer6.append(SpectralNorm(nn.Conv2d(curr_dim * 8, curr_dim * 8, 4, 2, 1)))
        layer6.append(nn.LeakyReLU(0.1))
        self.l6 = nn.Sequential(*layer6)

        layer7.append(SpectralNorm(nn.Conv2d(curr_dim * 8, curr_dim * 8, 4, 2, 1)))
        layer7.append(nn.LeakyReLU(0.1))
        self.l7 = nn.Sequential(*layer7)
        

        last.append(nn.Conv2d(curr_dim * 8, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn0 = Self_Attn(curr_dim * 2, 'relu')
        self.attn1 = Self_Attn(curr_dim * 4, 'relu')
        self.attn2 = Self_Attn(curr_dim * 8, 'relu')
        self.attn3 = Self_Attn(curr_dim * 8, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out,p0 = self.attn0(out) #[1, 256, 8, 8]
        out = self.l3(out)
        out,p1 = self.attn1(out) #[1, 256, 8, 8]
        out = self.l4(out)
        out = self.l5(out)
        out,p2 = self.attn2(out) #[1, 256, 8, 8]
        out = self.l6(out)
        out = self.l7(out)
        out,p3 = self.attn3(out) #[1, 256, 8, 8]
        out=self.last(out) #[1, 512, 4, 4]
        # return out.squeeze(), p1, p2 #[1, 1, 1, 1]
        return out.squeeze()
############################################################################################################################################
def resnet_block(in_channels, conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, kernel_size = 3, dilation = [1,1], bias=True):
    return ResnetBlock(in_channels,conv_layer, norm_layer, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels,conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, kernel_size = 3, dilation = [1,1], bias=True):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size=kernel_size, stride=1, dilation=dilation[0], bias=bias),
#             nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            # SpectralNorm(conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding = ((kernel_size-1)//2)*dilation[1], bias=bias)),
            conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding = ((kernel_size-1)//2)*dilation[1], bias=bias)
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out


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

class MSDilateBlockStyle(nn.Module):
    def __init__(self, in_channels,conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, kernel_size=3, dilation=[1,1,1,1], bias=True):
        super(MSDilateBlockStyle, self).__init__()
        self.conv1 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  conv_layer(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out
##############################UNetFace#########################

# def adaptive_instance_normalization_4D(content_feat, style_feat): # content_feat is ref feature, style is degradate feature
#     assert (content_feat.size()[:2] == style_feat.size()[:2])
#     size = content_feat.size()
#     style_mean, style_std = calc_mean_std_4D(style_feat)
#     content_mean, content_std = calc_mean_std_4D(content_feat)

#     normalized_feat = (content_feat - content_mean.expand(
#         size)) / content_std.expand(size)
#     return normalized_feat * style_std.expand(size) + style_mean.expand(size)


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
        # print('wwwwwwwwwwwwwwwwwww')
        # print(image.size())
        # print(noise.size())
        # print('############################')
        return image + self.weight * noise

class StyledUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,upsample=False):
        super().__init__()
        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
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
                nn.Upsample(scale_factor=2, mode='bilinear'),
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
        # self.ScaleModel2 = nn.Sequential(
        #     Blur(out_channel),
        #     nn.Conv2d(out_channel,out_channel,3, 1, 1),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        # )
        # self.ShiftModel2 = nn.Sequential(
        #     Blur(out_channel),
        #     nn.Conv2d(out_channel,out_channel,3, 1, 1),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        # )

    def forward(self, input, style):

        
        out = self.conv1(input)
#         out = self.noise1(out, noise)
        out = self.lrelu1(out)

        Shift1 = self.ShiftModel1(style)
        Scale1 = self.ScaleModel1(style)
        out = out * Scale1 + Shift1
        # print('eeeeeeeeeeeee')
        # print([out.size(), style.size()])
        # out = self.adain1(out, style)
        outup = self.convup(out)

#         out = self.conv2(out)
# #         out = self.noise2(out, noise)
#         out = self.lrelu2(out)

#         Shift2 = self.ShiftModel2(style)
#         Scale2 = self.ScaleModel2(style)
#         out = out * Scale2 + Shift2
# #         out = self.adain2(out, style)
        return outup
class UNetFace(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, ngf, 3, 1, 1)),
#             nn.Conv2d(3, ngf, 3, 1, 1),
#             nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),
            )
        self.stem1_1 = resnet_block(ngf, dilation=[3,3])
        self.stem1_2 = resnet_block(ngf, dilation=[3,3]) #ngf * 256 * 256

        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf*2, 3, 2, 1)),
#             nn.Conv2d(ngf, ngf*2, 3, 2, 1),
#             nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2),
            )
        self.stem2_1 = resnet_block(ngf*2, dilation=[3,3])
        self.stem2_2 = resnet_block(ngf*2, dilation=[3,3]) #(ngf*2) * 128 * 128

        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*2, ngf*4, 3, 2, 1)),
#             nn.Conv2d(ngf*2, ngf*4, 3, 2, 1),
#             nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2),
            )

        self.stem3_1 = resnet_block(ngf*4, dilation=[3,3])
        self.stem3_2 = resnet_block(ngf*4, dilation=[3,3]) #(ngf*4) * 64 * 64

        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*4, ngf*8, 3, 2, 1)),
#             nn.Conv2d(ngf*4, ngf*8, 3, 2, 1),
#             nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2),
            )

        self.stem4_1 = resnet_block(ngf*8, dilation=[3,3])
        self.stem4_2 = resnet_block(ngf*8, dilation=[3,3]) #(ngf*8) * 32 * 32
#         self.BlurInputConv1 = nn.Conv2d(3,3, 3,1,1, bias=False)


        ###stem 2

        self.b_conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, ngf, 3, 1, 1)),
#             nn.Conv2d(3, ngf, 5, 1, 2),
#             nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),
            )
        self.b_stem1_1 = resnet_block(ngf, dilation=[5,5])
        self.b_stem1_2 = resnet_block(ngf, dilation=[5,5]) #ngf * 256 * 256

        self.b_conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf*2, 3, 2, 1)),
#             nn.Conv2d(ngf, ngf*2, 5, 2, 2),
#             nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2),
            )
        self.b_stem2_1 = resnet_block(ngf*2, dilation=[5,5])
        self.b_stem2_2 = resnet_block(ngf*2, dilation=[5,5]) #(ngf*2) * 128 * 128

        self.b_conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*2, ngf*4, 3, 2, 1)),
#             nn.Conv2d(ngf*2, ngf*4, 5, 2, 2),
#             nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2),
            )

        self.b_stem3_1 = resnet_block(ngf*4, dilation=[5,5])
        self.b_stem3_2 = resnet_block(ngf*4, dilation=[5,5]) #(ngf*4) * 64 * 64

        self.b_conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*4, ngf*8, 3, 2, 1)),
#             nn.Conv2d(ngf*4, ngf*8, 5, 2, 2),
#             nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2),
            )

        self.b_stem4_1 = resnet_block(ngf*8, dilation=[5,5])
        self.b_stem4_2 = resnet_block(ngf*8, dilation=[5,5]) #(ngf*8) * 32 * 32
#         self.BlurInputConv2 = nn.Conv2d(3,3, 5,1,2, bias=False)

        ######################
        ######################

        self.MSDilate = MSDilateBlock(ngf*8, dilation = [4,3,2,1])  #(ngf*8) * 32 * 32


        # self.up0 = StyledUpBlock(ngf*8,ngf*)
        self.up1 = StyledUpBlock(ngf*8, ngf*4) #[6, 256, 64, 64]
        self.up2 = StyledUpBlock(ngf*4, ngf*2) #[6, 128, 128, 128]
        self.up3 = StyledUpBlock(ngf*2, ngf) #[6, 64, 256, 256]
        self.up4 = nn.Sequential( # 64
            nn.Upsample(scale_factor=2, mode='bilinear'),
            SpectralNorm(nn.Conv2d(ngf, ngf//2, 3, 1, 1)),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            UpResBlock(ngf//2),
            SpectralNorm(nn.Conv2d(ngf//2, 3, kernel_size=3, stride=1, padding=1)),
          
            nn.Tanh()
        )

        # n= np.zeros((7,7))
        # n[3,3] = 1
        # k = torch.from_numpy(scipy.ndimage.gaussian_filter(n,sigma=3))
        # print(k)
        # print(k.sum())

        # self.BlurInputConv.weight.data.copy_(k)
        # for param in self.BlurInputConv.parameters():
        #     param.requires_grad = False

    def forward(self,input):
#         inputblur = self.BlurInputConv1(input) # gaussian blur first
        # for i in range(input.size(0)):
        #     im1 = util.VisualFeature(inputblur[i])
        #     util.save_image(im1, './blurinput%s.png' % (i))
        # exit('wwwwwwwwwwwwwwwww')
        fea1 = self.stem1_2(self.stem1_1(self.conv1(input))) #ngf * 256 * 256
        fea2 = self.stem2_2(self.stem2_1(self.conv2(fea1))) #(ngf*2) * 128 * 128
        fea3 = self.stem3_2(self.stem3_1(self.conv3(fea2))) #(ngf*4) * 64 * 64
        fea4 = self.stem4_2(self.stem4_1(self.conv4(fea3))) #(ngf*8) * 32 * 32

#         b_inputblur = self.BlurInputConv2(input) # gaussian blur first
        b_fea1 = self.b_stem1_2(self.b_stem1_1(self.b_conv1(input))) #ngf * 256 * 256
        b_fea2 = self.b_stem2_2(self.b_stem2_1(self.b_conv2(b_fea1))) #(ngf*2) * 128 * 128
        b_fea3 = self.b_stem3_2(self.b_stem3_1(self.b_conv3(b_fea2))) #(ngf*4) * 64 * 64
        b_fea4 = self.b_stem4_2(self.b_stem4_1(self.b_conv4(b_fea3))) #(ngf*8) * 32 * 32

        # c_inputblur = self.BlurInputConv3(input) # gaussian blur first
        # c_fea1 = self.c_stem1_2(self.c_stem1_1(self.c_conv1(c_inputblur))) #ngf * 256 * 256
        # c_fea2 = self.c_stem2_2(self.c_stem2_1(self.c_conv2(c_fea1))) #(ngf*2) * 128 * 128
        # c_fea3 = self.c_stem3_2(self.c_stem3_1(self.c_conv3(c_fea2))) #(ngf*4) * 64 * 64
        # c_fea4 = self.c_stem4_2(self.c_stem4_1(self.c_conv4(c_fea3))) #(ngf*8) * 32 * 32
        fea4 = self.MSDilate(fea4, b_fea4) #(ngf*8) * 32 * 32 [1, 256, 128, 128]
#         fea4 = self.MSDilate(fea4)
#         noise64 = torch.randn(fea3.size(0),1, fea3.size(2),fea3.size(3)).type_as(input)
#         noise128 = torch.randn(fea2.size(0),1, fea2.size(2),fea2.size(3)).type_as(input)
#         noise256 = torch.randn(fea1.size(0),1, fea1.size(2),fea1.size(3)).type_as(input)
        fea_up1 = self.up1(fea4,fea3) #[6, 256, 64, 64]
        fea_up2 = self.up2(fea_up1,fea2) #[6, 128, 128, 128]
        fea_up3 = self.up3(fea_up2,fea1) #[6, 64, 256, 256]
        fea_up4 = self.up4(fea_up3) #[6, 3, 256, 256]
        return fea_up4
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
####
#第三版本，将updated feature与vgg feature直接串联，加入了part attention
###
class ConcatAttention(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=[1,1,1,1]):
        super().__init__()
        self.ConcatBlock = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel*2, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            # resnet_block(out_channel, dilation=dilation),
            MSDilateAttentionBlock(out_channel, dilation=dilation),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            # resnet_block(out_channel, dilation=dilation)
            MSDilateAttentionBlock(out_channel, dilation=dilation),
            # nn.LeakyReLU(0.2)
        )
    def forward(self, VggFeat, SwapVggFeat):
        inputs = torch.cat([VggFeat, SwapVggFeat], 1)
        return self.ConcatBlock(inputs)

class MSDilateAttentionBlock(nn.Module):
    def __init__(self, in_channels, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, kernel_size=3, dilation=[1,3,5,7], bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.conv2 =  convU(in_channels, in_channels//2,conv_layer, norm_layer, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  convU(in_channels, in_channels//2,conv_layer, norm_layer, kernel_size,dilation=dilation[2], bias=bias)
        # self.conv4 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)
    def forward(self, x):
        # print([x.size(), self.in_channels])
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        # conv4 = self.conv4(x)
        cat  = torch.cat([conv2, conv3], 1)
        # print([cat.size(), self.in_channels])
        out = self.convi(cat) + x
        return out

####
#第二版本，通过updated feature对vgg feature 进行仿射变换，加入了part attention
###
class UNetDictFace(nn.Module):
    def __init__(self, ngf=64, dictionary_path='/disk1/lxm/face_dict/DictionaryCenter512'):
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
            # print(f_256.size())
            # print(f_256[0:2].size())
            # f_256.requires_grad = False
            # f_256 = f_256[0:16]
            f_256_reshape = f_256.reshape(f_256.size(0),self.channel_sizes[0],self.part_sizes[j]//2,self.part_sizes[j]//2)
            max_256 = torch.max(torch.sqrt(compute_sum(torch.pow(f_256_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_256[i] = f_256_reshape #/ max_256
            
            f_128 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_128_center.npy'.format(i)), allow_pickle=True))
            # f_128 = f_128[0:16]
            f_128_reshape = f_128.reshape(f_128.size(0),self.channel_sizes[1],self.part_sizes[j]//4,self.part_sizes[j]//4)
            max_128 = torch.max(torch.sqrt(compute_sum(torch.pow(f_128_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_128[i] = f_128_reshape #/ max_128

            f_64 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_64_center.npy'.format(i)), allow_pickle=True))
            # f_64 = f_64[0:16]
            f_64_reshape = f_64.reshape(f_64.size(0),self.channel_sizes[2],self.part_sizes[j]//8,self.part_sizes[j]//8)
            max_64 = torch.max(torch.sqrt(compute_sum(torch.pow(f_64_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_64[i] = f_64_reshape #/ max_64

            f_32 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_32_center.npy'.format(i)), allow_pickle=True))
            # f_32 = f_32[0:16]
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
            # nn.Upsample(scale_factor=2, mode='bilinear'),
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
        # self.MaxList256 = []
        # self.MaxList128 = []
        # self.MaxList64 = []
        # self.MaxList32 = []
    
    def forward(self,input, part_locations):
        # self.MOA = torch.randn(2,1, 256,256).type_as(input)
        # self.NOA = torch.randn(2,1, 256,256).type_as(input)
        # self.LEA = torch.randn(2,1, 256,256).type_as(input)
        # self.REA = torch.randn(2,1, 256,256).type_as(input)
        

        VggFeatures = self.VggExtract(input)
        # for b in range(input.size(0)):
        b = 0
        UpdateVggFeatures = []
        for i, f_size in enumerate(self.feature_sizes):
            cur_feature = VggFeatures[i]
            update_feature = cur_feature.clone() #* 0 #这里skip到后面的只有部位特征
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
            LE_feature_resize = F.interpolate(LE_feature,(LE_Dict_feature.size(2),LE_Dict_feature.size(3)),mode='bilinear')
            RE_feature_resize = F.interpolate(RE_feature,(RE_Dict_feature.size(2),RE_Dict_feature.size(3)),mode='bilinear')
            NO_feature_resize = F.interpolate(NO_feature,(NO_Dict_feature.size(2),NO_Dict_feature.size(3)),mode='bilinear')
            MO_feature_resize = F.interpolate(MO_feature,(MO_Dict_feature.size(2),MO_Dict_feature.size(3)),mode='bilinear')

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


            # if i == 0:
            #     self.MOA = F.interpolate(MO_Attention, (256, 256))
            #     self.NOA = F.interpolate(NO_Attention, (256, 256))
            #     self.LEA = F.interpolate(LE_Attention, (256, 256))
            #     self.REA = F.interpolate(RE_Attention, (256, 256))

            # getattr(self, 'MaxList'+str(f_size)).append(LE_index)
            # getattr(self, 'MaxList'+str(f_size)).append(RE_index)
            # getattr(self, 'MaxList'+str(f_size)).append(NO_index)
            # getattr(self, 'MaxList'+str(f_size)).append(MO_index)

            # update_feature[:,:,le_location[1]:le_location[3],le_location[0]:le_location[2]] = LE_Swap_feature
            # update_feature[:,:,re_location[1]:re_location[3],re_location[0]:re_location[2]] = RE_Swap_feature
            # update_feature[:,:,no_location[1]:no_location[3],no_location[0]:no_location[2]] = NO_Swap_feature
            # update_feature[:,:,mo_location[1]:mo_location[3],mo_location[0]:mo_location[2]] = MO_Swap_feature
            
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
        
    
        return output#, self.MOA, self.NOA, self.LEA, self.REA, self.MaxList256, self.MaxList128, self.MaxList64, self.MaxList32 #+ out4 + out3 + out2 + out1
        #0 128 * 256 * 256
        #1 256 * 128 * 128
        #2 512 * 64 * 64
        #3 512 * 32 * 32

####
#第一版本，通过vgg feature对update feature 进行仿射变换
###
class UNetDictFaceV0(nn.Module):
    def __init__(self, ngf=64, dictionary_path='/disk1/lxm/face_dict/DictionaryCenter128'):
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
            # f_256.requires_grad = False
            f_256_reshape = f_256.reshape(f_256.size(0),self.channel_sizes[0],self.part_sizes[j]//2,self.part_sizes[j]//2)
            max_256 = torch.max(torch.sqrt(compute_sum(torch.pow(f_256_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_256[i] = f_256_reshape / max_256
            
            f_128 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_128_center.npy'.format(i)), allow_pickle=True))
            f_128_reshape = f_128.reshape(f_128.size(0),self.channel_sizes[1],self.part_sizes[j]//4,self.part_sizes[j]//4)
            max_128 = torch.max(torch.sqrt(compute_sum(torch.pow(f_128_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_128[i] = f_128_reshape / max_128

            f_64 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_64_center.npy'.format(i)), allow_pickle=True))
            f_64_reshape = f_64.reshape(f_64.size(0),self.channel_sizes[2],self.part_sizes[j]//8,self.part_sizes[j]//8)
            max_64 = torch.max(torch.sqrt(compute_sum(torch.pow(f_64_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_64[i] = f_64_reshape / max_64

            f_32 = torch.from_numpy(np.load(os.path.join(dictionary_path, '{}_32_center.npy'.format(i)), allow_pickle=True))
            f_32_reshape = f_32.reshape(f_32.size(0),self.channel_sizes[3],self.part_sizes[j]//16,self.part_sizes[j]//16)
            max_32 = torch.max(torch.sqrt(compute_sum(torch.pow(f_32_reshape, 2), axis=[1, 2, 3], keepdim=True)),torch.FloatTensor([1e-4]))
            self.Dict_32[i] = f_32_reshape / max_32
            
        #norm
        
        self.VggExtract = VGGFeat()

        
        ######################
        self.MSDilate = MSDilateBlock(ngf*8, dilation = [4,3,2,1])  #

        self.up0 = StyledUpBlock(ngf*8,ngf*8)
        self.up1 = StyledUpBlock(ngf*8, ngf*4) #
        self.up2 = StyledUpBlock(ngf*4, ngf*2) #
        self.up3 = nn.Sequential( # 128
            nn.Upsample(scale_factor=2, mode='bilinear'),
            SpectralNorm(nn.Conv2d(ngf*2, ngf, 3, 1, 1)),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            UpResBlock(ngf),
            UpResBlock(ngf),
            SpectralNorm(nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)),
            nn.Tanh()
        )
        self.to_rgb0 = ToRGB(ngf*8)
        self.to_rgb1 = ToRGB(ngf*4)
        self.to_rgb2 = ToRGB(ngf*2)

        # for param in self.BlurInputConv.parameters():
        #     param.requires_grad = False
    
    def forward(self,input, part_locations):
        VggFeatures = self.VggExtract(input)
        # for b in range(input.size(0)):
        b = 0
        UpdateVggFeatures = []
        for i, f_size in enumerate(self.feature_sizes):
            cur_feature = VggFeatures[i]
            update_feature = cur_feature.clone()
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
            LE_feature_resize = F.interpolate(LE_feature,(LE_Dict_feature.size(2),LE_Dict_feature.size(3)),mode='bilinear')
            RE_feature_resize = F.interpolate(RE_feature,(RE_Dict_feature.size(2),RE_Dict_feature.size(3)),mode='bilinear')
            NO_feature_resize = F.interpolate(NO_feature,(NO_Dict_feature.size(2),NO_Dict_feature.size(3)),mode='bilinear')
            MO_feature_resize = F.interpolate(MO_feature,(MO_Dict_feature.size(2),MO_Dict_feature.size(3)),mode='bilinear')
            
            LE_Dict_feature_norm = adaptive_instance_normalization_4D(LE_Dict_feature, LE_feature_resize)
            RE_Dict_feature_norm = adaptive_instance_normalization_4D(RE_Dict_feature, RE_feature_resize)
            NO_Dict_feature_norm = adaptive_instance_normalization_4D(NO_Dict_feature, NO_feature_resize)
            MO_Dict_feature_norm = adaptive_instance_normalization_4D(MO_Dict_feature, MO_feature_resize)

            LE_score = F.conv2d(LE_feature_resize, LE_Dict_feature_norm)
            LE_score = F.softmax(LE_score.view(-1),dim=0)
            LE_index = torch.argmax(LE_score)
            LE_Swap_feature = F.interpolate(LE_Dict_feature_norm[LE_index:LE_index+1], (LE_feature.size(2), LE_feature.size(3)))
            update_feature[:,:,le_location[1]:le_location[3],le_location[0]:le_location[2]] = LE_Swap_feature

            RE_score = F.conv2d(RE_feature_resize, RE_Dict_feature_norm)
            RE_score = F.softmax(RE_score.view(-1),dim=0)
            RE_index = torch.argmax(RE_score)
            RE_Swap_feature = F.interpolate(RE_Dict_feature_norm[RE_index:RE_index+1], (RE_feature.size(2), RE_feature.size(3)))
            update_feature[:,:,re_location[1]:re_location[3],re_location[0]:re_location[2]] = RE_Swap_feature
            
            NO_score = F.conv2d(NO_feature_resize, NO_Dict_feature_norm)
            NO_score = F.softmax(NO_score.view(-1),dim=0)
            NO_index = torch.argmax(NO_score)
            NO_Swap_feature = F.interpolate(NO_Dict_feature_norm[NO_index:NO_index+1], (NO_feature.size(2), NO_feature.size(3)))
            update_feature[:,:,no_location[1]:no_location[3],no_location[0]:no_location[2]] = NO_Swap_feature

            MO_score = F.conv2d(MO_feature_resize, MO_Dict_feature_norm)
            MO_score = F.softmax(MO_score.view(-1),dim=0)
            MO_index = torch.argmax(MO_score)
            MO_Swap_feature = F.interpolate(MO_Dict_feature_norm[MO_index:MO_index+1], (MO_feature.size(2), MO_feature.size(3)))
            update_feature[:,:,mo_location[1]:mo_location[3],mo_location[0]:mo_location[2]] = MO_Swap_feature

            UpdateVggFeatures.append(update_feature) 
        
        fea_fusion = self.MSDilate(UpdateVggFeatures[3])
        fea_up0 = self.up0(fea_fusion, VggFeatures[2])
        out1 = F.interpolate(fea_up0,(512,512))
        out1 = self.to_rgb0(out1)

        fea_up1 = self.up1(fea_up0, VggFeatures[1]) #
        out2 = F.interpolate(fea_up1,(512,512))
        out2 = self.to_rgb1(out2)

        fea_up2 = self.up2(fea_up1,VggFeatures[0]) #
        out3 = F.interpolate(fea_up2,(512,512))
        out3 = self.to_rgb2(out3)

        output = self.up3(fea_up2) #

        return output #+ out3 + out2 + out1
        #0 128 * 256 * 256
        #1 256 * 128 * 128
        #2 512 * 64 * 64
        #3 512 * 32 * 32


       
#         fea1 = self.stem1_2(self.stem1_1(self.conv1(input))) #ngf * 256 * 256
#         fea2 = self.stem2_2(self.stem2_1(self.conv2(fea1))) #(ngf*2) * 128 * 128
#         fea3 = self.stem3_2(self.stem3_1(self.conv3(fea2))) #(ngf*4) * 64 * 64
#         fea4 = self.stem4_2(self.stem4_1(self.conv4(fea3))) #(ngf*8) * 32 * 32

# #         b_inputblur = self.BlurInputConv2(input) # gaussian blur first
#         b_fea1 = self.b_stem1_2(self.b_stem1_1(self.b_conv1(input))) #ngf * 256 * 256
#         b_fea2 = self.b_stem2_2(self.b_stem2_1(self.b_conv2(b_fea1))) #(ngf*2) * 128 * 128
#         b_fea3 = self.b_stem3_2(self.b_stem3_1(self.b_conv3(b_fea2))) #(ngf*4) * 64 * 64
#         b_fea4 = self.b_stem4_2(self.b_stem4_1(self.b_conv4(b_fea3))) #(ngf*8) * 32 * 32

#         fea4 = self.MSDilate(fea4, b_fea4) #(ngf*8) * 32 * 32 [1, 256, 128, 128]
# #         fea4 = self.MSDilate(fea4)
# #         noise64 = torch.randn(fea3.size(0),1, fea3.size(2),fea3.size(3)).type_as(input)
# #         noise128 = torch.randn(fea2.size(0),1, fea2.size(2),fea2.size(3)).type_as(input)
# #         noise256 = torch.randn(fea1.size(0),1, fea1.size(2),fea1.size(3)).type_as(input)

#         fea_up1 = self.up1(fea4,fea3) #[6, 256, 64, 64]
#         fea_up2 = self.up2(fea_up1,fea2) #[6, 128, 128, 128]
#         fea_up3 = self.up3(fea_up2,fea1) #[6, 64, 256, 256]
#         fea_up4 = self.up4(fea_up3) #[6, 3, 256, 256]
#         return fea_up4


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

# class MSDilateAttentionBlock(nn.Module):
#     def __init__(self, in_channels,conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, kernel_size=3, dilation=[1,3,5,7], bias=True):
#         super(MSDilateAttentionBlock, self).__init__()
#         # self.conv1 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[0], bias=bias)
#         self.conv2 =  convU(in_channels, in_channels//2,conv_layer, norm_layer, kernel_size,dilation=dilation[1], bias=bias)
#         self.conv3 =  convU(in_channels, in_channels//2,conv_layer, norm_layer, kernel_size,dilation=dilation[2], bias=bias)
#         # self.conv4 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[3], bias=bias)
#         self.convi =  conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)
#     def forward(self, x):
#         # conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         # conv4 = self.conv4(x)
#         cat  = torch.cat([conv2, conv3], 1)
#         out = self.convi(cat) + x
#         return out

class UpDilateResBlock(nn.Module):
    def __init__(self, dim, dilation=[5,3] ):
        super(UpDilateResBlock, self).__init__()
        self.Model = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[0], dilation[0]),
            # nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[1], dilation[1]),
            nn.LeakyReLU(0.2,True),
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

###########################################################
###Style Face
###########################################################
class StyledFaceUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,upsample=True):
        super().__init__()
        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                EqualConv2d(in_channel, out_channel, kernel_size, padding=padding),
                Blur(out_channel),
            )
        else:
            self.conv1 = EqualConv2d(
                in_channel, out_channel, kernel_size, padding=padding
            )
        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)

        # self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        # self.noise2 = equal_lr(NoiseInjection(out_channel))
        # self.adain2 = AdaptiveInstanceNorm(out_channel)
        # self.lrelu2 = nn.LeakyReLU(0.2)

        self.ScaleModel1 = nn.Sequential(
            Blur(out_channel),
            SpectralNorm(nn.Conv2d(out_channel,out_channel,3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
        )
        self.ShiftModel1 = nn.Sequential(
            Blur(out_channel),
            SpectralNorm(nn.Conv2d(out_channel,out_channel,3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
        )


    def forward(self, input, style, noise):

        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)

        Shift1 = self.ShiftModel1(style)
        Scale1 = self.ScaleModel1(style)
        out = out * Scale1 + Shift1
        out = self.adain1(out, style)

        # out = self.conv2(out)
        # out = self.noise2(out, noise)
        # out = self.lrelu2(out)

        # Shift2 = self.ShiftModel2(style)
        # Scale2 = self.ScaleModel2(style)
        # out = out * Scale2 + Shift2
#         out = self.adain2(out, style)

        return out
class ChannelAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Sequential(
            # SpectralNorm(nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True)),
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            # SpectralNorm(nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True)),
            nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        y = self.avg_pool(x)
        y = self.conv_ca(y)
        return x * y

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, in_channel, kernel_size=3, res_scale=1):

        super(ResidualChannelAttentionBlock, self).__init__()
        modules_body = []
        for i in range(2):
            # modules_body.append(SpectralNorm(nn.Conv2d(in_channel, in_channel, kernel_size, padding=1)))
            modules_body.append(nn.Conv2d(in_channel, in_channel, kernel_size, padding=1))
            modules_body.append(nn.LeakyReLU(0.2))
        modules_body.append(ChannelAttentionLayer(in_channel))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class RCABGroup(nn.Module):
    def __init__(self, in_channel, n_blocks=6):
        super(RCABGroup, self).__init__()
        modules_body = []
        for _ in range(n_blocks):
            modules_body.append(ResidualChannelAttentionBlock(in_channel))
        self.body = nn.Sequential(*modules_body)
    
    def forward(self,x):
        res = self.body(x)
        return x #+ res


class StyleFace(nn.Module):
    def __init__(self, ngf=64):
        super(StyleFace, self).__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, ngf, 3, 1, 1)),
#             nn.Conv2d(3, ngf, 3, 1, 1),
#             nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),
            )
        self.stem1_1 = resnet_block(ngf, dilation=[3,3])
        self.stem1_2 = resnet_block(ngf, dilation=[3,3]) #ngf * 256 * 256

        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf*2, 3, 2, 1)),
#             nn.Conv2d(ngf, ngf*2, 3, 2, 1),
#             nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2),
            )
        self.stem2_1 = resnet_block(ngf*2, dilation=[3,3])
        self.stem2_2 = resnet_block(ngf*2, dilation=[3,3]) #(ngf*2) * 128 * 128

        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*2, ngf*4, 3, 2, 1)),
#             nn.Conv2d(ngf*2, ngf*4, 3, 2, 1),
#             nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2),
            )

        self.stem3_1 = resnet_block(ngf*4, dilation=[3,3])
        self.stem3_2 = resnet_block(ngf*4, dilation=[3,3]) #(ngf*4) * 64 * 64

        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*4, ngf*8, 3, 2, 1)),
#             nn.Conv2d(ngf*4, ngf*8, 3, 2, 1),
#             nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2),
            )

        self.stem4_1 = resnet_block(ngf*8, dilation=[3,3])
        self.stem4_2 = resnet_block(ngf*8, dilation=[3,3]) #(ngf*8) * 32 * 32
#         self.BlurInputConv1 = nn.Conv2d(3,3, 3,1,1, bias=False)

        ######################
        
        self.MSDilate = MSDilateBlockStyle(ngf*8, dilation = [4,3,2,1])  #(ngf*8) * 32 * 32

        self.center_block = RCABGroup(ngf*8,6)

        # self.up0 = StyledUpBlock(ngf*8,ngf*)
        self.up1 = StyledFaceUpBlock(ngf*8, ngf*4) #[6, 256, 64, 64]
        self.up2 = StyledFaceUpBlock(ngf*4, ngf*2) #[6, 128, 128, 128]
        self.up3 = StyledFaceUpBlock(ngf*2, ngf) #[6, 64, 256, 256]
        self.up4 = nn.Sequential( # 64
            nn.Upsample(scale_factor=2, mode='bilinear'),
            SpectralNorm(nn.Conv2d(ngf, ngf//2, 3, 1, 1)),
            # EqualConv2d(ngf, ngf//2, 3, padding=1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            UpResBlock(ngf//2),
            SpectralNorm(nn.Conv2d(ngf//2, 3, kernel_size=3, stride=1, padding=1)),
            # EqualConv2d(ngf//2, 3, 3, padding=1),
            nn.Tanh()
        )

        # n= np.zeros((7,7))
        # n[3,3] = 1
        # k = torch.from_numpy(scipy.ndimage.gaussian_filter(n,sigma=3))
        # print(k)
        # print(k.sum())
        
        
        # self.BlurInputConv.weight.data.copy_(k)
        # for param in self.BlurInputConv.parameters():
        #     param.requires_grad = False

        
        

    def forward(self,input):
#         inputblur = self.BlurInputConv1(input) # gaussian blur first
        # for i in range(input.size(0)):
        #     im1 = util.VisualFeature(inputblur[i])
        #     util.save_image(im1, './blurinput%s.png' % (i))
        # exit('wwwwwwwwwwwwwwwww')
        fea1 = self.stem1_2(self.stem1_1(self.conv1(input))) #ngf * 256 * 256
        fea2 = self.stem2_2(self.stem2_1(self.conv2(fea1))) #(ngf*2) * 128 * 128
        fea3 = self.stem3_2(self.stem3_1(self.conv3(fea2))) #(ngf*4) * 64 * 64
        fea4 = self.stem4_2(self.stem4_1(self.conv4(fea3))) #(ngf*8) * 32 * 32

        
        # fea4 = self.MSDilate(fea4, b_fea4) #(ngf*8) * 32 * 32 [1, 256, 128, 128]
        fea4 = self.MSDilate(fea4)
        
        noise64 = torch.randn(fea3.size(0),1, fea3.size(2),fea3.size(3)).type_as(input)
        noise128 = torch.randn(fea2.size(0),1, fea2.size(2),fea2.size(3)).type_as(input)
        noise256 = torch.randn(fea1.size(0),1, fea1.size(2),fea1.size(3)).type_as(input)
        fea_rca = self.center_block(fea4)
        
        fea_up1 = self.up1(fea_rca,fea3,noise64) #[6, 256, 64, 64]
        fea_up2 = self.up2(fea_up1,fea2,noise128) #[6, 128, 128, 128]
        fea_up3 = self.up3(fea_up2,fea1,noise256) #[6, 64, 256, 256]
        fea_up4 = self.up4(fea_up3) #[6, 3, 256, 256]
        return fea_up4



class StyleAndContentLoss(nn.Module):
    def __init__(self):
        super(StyleAndContentLoss, self).__init__()
        self.vgg = VggClassNet(['0','5','10','19'])
#         self.vgg = VggClassNet(['5','10','19'])
        self.loss = nn.MSELoss(size_average=False)
        self.register_parameter("RGB_mean", nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)))
        self.register_parameter("RGB_std", nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)))

        # self.weights = [1.0,  1.0 / 2, 1.0 / 4, 1.0 / 8 ]
        self.weights = [1.0 / 8,  1.0 / 4, 1.0 / 2, 1.0]

    def forward(self, x, y):
#         x = F.interpolate(input=x, size=(256, 256), mode='bilinear')
#         y = F.interpolate(input=y, size=(256, 256), mode='bilinear')
        x = (x+1)/2.0   
        y = (y+1)/2.0

        Norm_X = (x - self.RGB_mean) / self.RGB_std
        Norm_Y = (y - self.RGB_mean) / self.RGB_std
        restored_feature = self.vgg(Norm_X)
        gd_feature = self.vgg(Norm_Y)
        style_loss = 0
        content_loss = 0
        i = 0
        for f1, f2 in zip(restored_feature, gd_feature):
            b, c, h, w = f1.size()
            content_loss += self.loss(f1, f2) * self.weights[i] / (b * c * h * w)
            i = i + 1
            f1T = f1.view(b, c, h * w)
            f2T = f2.view(b, c, h * w)
            f1G = torch.bmm(f1T, f1T.transpose(1, 2))
            f2G = torch.bmm(f2T, f2T.transpose(1, 2))
            
            style_loss = style_loss + torch.mean((f1G - f2G) ** 2) / (b * c * h * w)

        return style_loss, content_loss

class TVLossImg(nn.Module):
    def __init__(self):
        super(TVLossImg, self).__init__()
    def forward(self, x):
        # print(self.ori_xy.sum())
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return (h_tv/count_h+w_tv/count_w)#/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


# Define a resnet block
class ResnetBlockSimple(nn.Module):
    def __init__(self, dim, padding_type, use_dropout=False, use_bias=True):
        super(ResnetBlockSimple, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       nn.LeakyReLU(0.2)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)

        return out
#######################################################################
#######for SN discriminator
#######################################################################

class SNResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SNResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class SNFirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SNFirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.LeakyReLU(0.2),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        # a = self.model(x) #6*64*128*128
        # b = self.bypass(x)
        # print(a.size())
        # print(b.size())
        return self.model(x) + self.bypass(x)

class SNDiscriminator(nn.Module):
    def __init__(self, DISC_SIZE = 128):
        super(SNDiscriminator, self).__init__()
        Res1 = SNFirstResBlockDiscriminator(3, 64) # 128
        Res2 = SNResBlockDiscriminator(64, 128, stride=2) # 64
        Res3 = SNResBlockDiscriminator(128, 256, stride=2) # 32
        Res4 = SNResBlockDiscriminator(256, 512, stride=2) # 16
        Res5 = SNResBlockDiscriminator(512, 512, stride=2) # 8

        self.model = nn.Sequential(
                Res1,
                Res2,
                Res3,
                Res4,
                Res5,
                nn.ReLU(),
                nn.AvgPool2d(8),
            ) #output 512*1*1
        self.fc = nn.Linear(512, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1,512))


class NScaleSNDiscriminator(nn.Module):
    def __init__(self, dim = 64, Scale = 512):
        super(NScaleSNDiscriminator, self).__init__()
        self.model = []
        self.model.append(SNFirstResBlockDiscriminator(3, dim * 1))
        cur_dim = dim
        BlockNum = 5 - int(math.log(512//Scale,2))
        for i in range(BlockNum):
            self.model.append(SNResBlockDiscriminator(cur_dim, min(cur_dim*2,512), stride=2))
            cur_dim = cur_dim * 2
            cur_dim = min(cur_dim,512)
        self.model.append(nn.AvgPool2d(8 * Scale // 512 ))
        self.models = nn.Sequential(*self.model)
        self.fc = nn.Linear(cur_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.models(x).view(-1,512))

class MultiScaleSNDiscriminator(nn.Module):
    def __init__(self, Scales = [512,256,128]):
        super(MultiScaleSNDiscriminator, self).__init__()
        self.D_pools = nn.ModuleList()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        for scale in Scales:
            netD  = NScaleSNDiscriminator(Scale = scale)
            self.D_pools.append(netD)
    
    def forward(self,input):
        results = []
        for netD in self.D_pools:
            output = netD(input) 
            results.append(output)
            # Downsample input
            input = self.downsample(input)
        return results
    
    
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)
class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out
class StyleDiscriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=True):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
#                 ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
#                 make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=7, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
#         print(out.size())
        out = self.linear(out)
#         print(out.size())

        return out

##################################################################################
##multiscale discriminator
##################################################################################
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_ch, base_ch=64, n_layers=3, norm_type='none', relu_type='LeakyReLU', num_D=4, ref_ch=None):
        super().__init__()
        self.D_pool = nn.ModuleList()
        for i in range(num_D):
            netD = NLayerDiscriminator(input_ch, base_ch, depth=n_layers, norm_type=norm_type, relu_type=relu_type, ref_ch=ref_ch)
            self.D_pool.append(netD)

        # self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bicubic')

    def forward(self, input, return_feat=False):
        results = []
        for netd in self.D_pool:
            output = netd(input, return_feat) 
            results.append(output)
            # Downsample input
            input = self.downsample(input)
        return results


class NLayerDiscriminator(nn.Module):
    def __init__(self,
            input_ch = 3,
            base_ch = 64,
            max_ch = 1024,
            depth = 4,
            norm_type = 'none',
            relu_type = 'LeakyReLU',
            ref_ch = None,
            ):
        super().__init__()

        nargs = {'norm_type': norm_type, 'relu_type': relu_type, 'ref_ch': ref_ch}
        self.norm_type = norm_type
        self.ref_ch = ref_ch
        self.input_ch = input_ch

        self.model = []
        self.model.append(ConvLayer(input_ch, base_ch, norm_type='none', relu_type=relu_type))
        for i in range(depth):
            cin  = min(base_ch * 2**(i), max_ch)
            cout = min(base_ch * 2**(i+1), max_ch)
            self.model.append(ConvLayer(cin, cout, scale='down', **nargs))
        self.model = nn.Sequential(*self.model)
        self.score_out = ConvLayer(cout, 1, use_pad=False)

    def forward(self, x, return_feat=False):
        if self.norm_type == 'spade':
            assert x.shape[1] == self.input_ch + self.ref_ch, 'channels == input_ch + ref_ch' 
            x, ref = x[:, :self.input_ch], x[:, self.input_ch:]

        ret_feats = []
        for idx, m in enumerate(self.model):
            if m.norm_type == 'spade':
                m.set_ref(ref)
            x = m(x)
            ret_feats.append(x)
        x = self.score_out(x)
        if return_feat:
            return x, ret_feats
        else:
            return x

if __name__ == '__main__':
    from torchvision import models
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = VGGFaceRoIAlignAdaIN().to(device)

    # summary(model,[(1,3,256,256),(1,3,256,256),(1,1,256,256),(1,68,2),(1,68,2),(1,4,4)])
    # model = FeatureExDilatedResNet()
    # summary(model,(1,3,256,256))

    model = UNetFace()
    summary(model, (3,256,256))


    #in resblock padding should meet
    #p = |（d*(k-1)-1） /2 |
    #
    #
