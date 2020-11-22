#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import os
import numpy as np
import math
import sys
import pdb

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from lenet import LeNet5Half
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import resnet
from datalmdb import DataLmdb
from mfn import MfnModel
from mfn_mini import MfnModelMini

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.02, help='learning rate')
parser.add_argument('--lr_S', type=float, default=0.1, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--oh', type=float, default=0.5, help='one hot loss')
parser.add_argument('--ie', type=float, default=20, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default='./')

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True 

accr = 0
accr_best = 0

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(opt.channels, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
        
generator = Generator().cuda()
    
num_classes = 796
teacher = MfnModel(n_class=num_classes).cuda()
teacher.load_state_dict( torch.load('mfn_org.pth') )
teacher.eval()
net = MfnModelMini(n_class=num_classes).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()

#teacher = nn.DataParallel(teacher)
#net = nn.DataParallel(net)
#generator = nn.DataParallel(generator)

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl

data_test_loader = torch.utils.data.DataLoader(DataLmdb("/kaggle/working/Valid-Low_lmdb", db_size=7939, crop_size=128, flip=False, scale=0.00390625, random=False),
        batch_size=256, shuffle=False)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
optimizer_S = torch.optim.SGD(net.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 800:
        lr = learing_rate
    elif epoch < 1600:
        lr = 0.1*learing_rate
    else:
        lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    total_correct = 0
    avg_loss = 0.0

    adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

    for i in range(1):
        net.train()
        z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()        
        gen_imgs = generator(z)
        outputs_T, features_T = teacher(gen_imgs, out_feature=True)   
        pred = outputs_T.data.max(1)[1]
        loss_activation = -features_T.abs().mean()
        loss_one_hot = criterion(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
        loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach()) 
        loss += loss_kd
        loss.backward()
        optimizer_G.step()
        optimizer_S.step() 
        if i == 1:
            print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))
            
    with torch.no_grad():
        len_data_test = 7936
        for i, (images, labels) in enumerate(data_test_loader):
            if 796 in labels:
                print('===error', i)
                len_data_test -= len(labels)
            
            images = images.cuda()
            labels = labels.cuda()
            net.eval()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    print('===========', avg_loss, total_correct)
    avg_loss /= float(len_data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss, float(total_correct) / len_data_test))
    accr = round(float(total_correct) / len_data_test, 4)
    if accr > accr_best:
        torch.save(net,opt.output_dir + 'student')
        accr_best = accr
