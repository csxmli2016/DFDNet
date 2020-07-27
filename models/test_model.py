from .base_model import BaseModel
from . import networks
import torch
import numpy as np
import torchvision.transforms as transforms
import PIL

import torch.nn.functional as F

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser.set_defaults(dataset_mode='aligned')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')
        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['fake_A','real_A']
        self.model_names = ['G']

        self.netG = networks.define_G('UNetDictFace',self.gpu_ids)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device) #degraded img
        self.real_C = input['C'].to(self.device) #groundtruth
        self.image_paths = input['A_paths']
        self.Part_locations = input['Part_locations']

    def forward(self):
        
        self.fake_A = self.netG(self.real_A, self.Part_locations) #
        # try:
        #     self.fake_A = self.netG(self.real_A, self.Part_locations) #生成图
        # except:
        #     self.fake_A = self.real_A
