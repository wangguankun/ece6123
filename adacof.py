import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from adacofmodel import adacof as adacof
import sys
from torch.nn import functional as F
from utility import CharbonnierFunc, moduleNormalize

class UNet():
  def __init__(self):
    self.conv1a = nn.Conv2d(6, 32, 3, padding=1)
    self.conv1b = nn.Conv2d(32, 32, 3, padding=1)

    self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
    self.conv2b = nn.Conv2d(64, 64, 3, padding=1)

    self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
    self.conv3b = nn.Conv2d(128, 128, 3, padding=1)

    self.conv4a = nn.Conv2d(128, 256, 3, padding=1)
    self.conv4b = nn.Conv2d(256, 256, 3, padding=1)

    self.conv5a = nn.Conv2d(256, 512, 3, padding=1)
    self.conv5b = nn.Conv2d(512, 512, 3, padding=1)

    self.conv6 = nn.Conv2d(512, 512, 3, padding=1)

    self.conv7a = nn.Conv2d(512, 256, 3, padding=1)
    self.conv7b = nn.Conv2d(256, 256, 3, padding=1)

    self.conv8a = nn.Conv2d(256, 128, 3, padding=1)
    self.conv8b = nn.Conv2d(128, 128, 3, padding=1)

    self.conv9a = nn.Conv2d(128, 64, 3, padding=1)
    self.conv9b = nn.Conv2d(64, 64, 3, padding=1)

    self.conv10 = nn.Conv2d(64, 64, 3, padding=1)

    self.pool = nn.AvgPool2d((2, 2))
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    self.leaky_relu = nn.LeakyReLU()

    self.conv_offset_a = nn.Conv2d(64, 64, 3, padding=1)
    self.conv_offset_b = nn.Conv2d(64, 1, 3, padding=1)

    self.conv_weight_a = nn.Conv2d(64, 64, 3, padding=1)
    self.conv_weight_b = nn.Conv2d(64, 1, 3, padding=1)
    self.softmax = nn.Softmax(dim=1)

    self.conva = nn.Conv2d(64, 64, 3, padding=1)
    self.convb = nn.Conv2d(64, 1, 3, padding=1)
    self.sigmoid = nn.Sigmoid(dim=1)

    self.kernel_pad = int(())

class AdaCoFNet(torch.nn.Module):
    def __init__(self, args):
        super(AdaCoFNet, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h), mode='reflect')
            frame2 = F.pad(frame2, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0), mode='reflect')
            frame2 = F.pad(frame2, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True
        Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion = self.get_kernel(moduleNormalize(frame0), moduleNormalize(frame2))

        tensorAdaCoF1 = self.moduleAdaCoF(self.modulePad(frame0), Weight1, Alpha1, Beta1, self.dilation)
        tensorAdaCoF2 = self.moduleAdaCoF(self.modulePad(frame2), Weight2, Alpha2, Beta2, self.dilation)

        frame1 = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2
        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]

        if self.training:
            # Smoothness Terms
            m_Alpha1 = torch.mean(Weight1 * Alpha1, dim=1, keepdim=True)
            m_Alpha2 = torch.mean(Weight2 * Alpha2, dim=1, keepdim=True)
            m_Beta1 = torch.mean(Weight1 * Beta1, dim=1, keepdim=True)
            m_Beta2 = torch.mean(Weight2 * Beta2, dim=1, keepdim=True)

            g_Alpha1 = CharbonnierFunc(m_Alpha1[:, :, :, :-1] - m_Alpha1[:, :, :, 1:]) + CharbonnierFunc(m_Alpha1[:, :, :-1, :] - m_Alpha1[:, :, 1:, :])
            g_Beta1 = CharbonnierFunc(m_Beta1[:, :, :, :-1] - m_Beta1[:, :, :, 1:]) + CharbonnierFunc(m_Beta1[:, :, :-1, :] - m_Beta1[:, :, 1:, :])
            g_Alpha2 = CharbonnierFunc(m_Alpha2[:, :, :, :-1] - m_Alpha2[:, :, :, 1:]) + CharbonnierFunc(m_Alpha2[:, :, :-1, :] - m_Alpha2[:, :, 1:, :])
            g_Beta2 = CharbonnierFunc(m_Beta2[:, :, :, :-1] - m_Beta2[:, :, :, 1:]) + CharbonnierFunc(m_Beta2[:, :, :-1, :] - m_Beta2[:, :, 1:, :])
            g_Occlusion = CharbonnierFunc(Occlusion[:, :, :, :-1] - Occlusion[:, :, :, 1:]) + CharbonnierFunc(Occlusion[:, :, :-1, :] - Occlusion[:, :, 1:, :])

            g_Spatial = g_Alpha1 + g_Beta1 + g_Alpha2 + g_Beta2

            return {'frame1': frame1, 'g_Spatial': g_Spatial, 'g_Occlusion': g_Occlusion}
        else:
            return frame1