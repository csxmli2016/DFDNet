import torch.nn as nn
import torch
import torch.nn.functional as F


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class hinge_loss(nn.Module):
    def __init__(self):
        super(hinge_loss, self).__init__()

    def forward(self, dis_fake, dis_real):
        loss_real = torch.mean(F.relu(1. - dis_real))
        loss_fake = torch.mean(F.relu(1. + dis_fake))
        return loss_real + loss_fake


class hinge_loss_G(nn.Module):
    def __init__(self):
        super(hinge_loss_G, self).__init__()

    def forward(self, dis_fake):
        loss_fake = -torch.mean(dis_fake)
        return loss_fake