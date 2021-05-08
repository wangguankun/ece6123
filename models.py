import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import cv2
import numpy as np
from loadimages import load_images
import random

class FlowNetS(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(6, 64, 7, padding=3)
      self.batch_norm1 = nn.BatchNorm2d(num_features = 64)
      self.activation1 = nn.LeakyReLU()

      self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
      self.batch_norm2 = nn.BatchNorm2d(num_features = 128)
      self.activation2 = nn.LeakyReLU()

      self.conv3 = nn.Conv2d(128, 256, 5, padding=2)
      self.batch_norm3 = nn.BatchNorm2d(num_features = 256)
      self.activation3 = nn.LeakyReLU()

      self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
      self.batch_norm4 = nn.BatchNorm2d(num_features = 256)
      self.activation4 = nn.LeakyReLU()

      self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
      self.batch_norm5 = nn.BatchNorm2d(num_features = 512)
      self.activation5 = nn.LeakyReLU()

      self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
      self.batch_norm6 = nn.BatchNorm2d(num_features = 512)
      self.activation6 = nn.LeakyReLU()

      self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
      self.batch_norm7 = nn.BatchNorm2d(num_features = 512)
      self.activation7 = nn.LeakyReLU()

      self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
      self.batch_norm8 = nn.BatchNorm2d(num_features = 512)
      self.activation8 = nn.LeakyReLU()

      self.conv9 = nn.Conv2d(512, 1024, 3, padding=1)
      self.batch_norm9 = nn.BatchNorm2d(num_features = 1024)
      self.activation9 = nn.LeakyReLU()

      self.conv10 = nn.Conv2d(1024, 512, 1, padding=0)
      self.batch_norm10 = nn.BatchNorm2d(num_features = 512)
      self.activation10 = nn.LeakyReLU()

      self.conv11 = nn.Conv2d(1024, 256, 1, padding=0)
      self.batch_norm11 = nn.BatchNorm2d(num_features = 256)
      self.activation11 = nn.LeakyReLU()
      self.flow1 = nn.Conv2d(1024, 3, 5, padding=2)
      self.batch_norm_flow1 = nn.BatchNorm2d(num_features = 3)
      self.flow_activation1 = nn.LeakyReLU()


      self.conv12 = nn.Conv2d(771, 128, 1, padding=0)
      self.batch_norm12 = nn.BatchNorm2d(num_features = 128)
      self.activation12 = nn.LeakyReLU()
      self.flow2 = nn.Conv2d(771, 3, 5, padding=2)
      self.batch_norm_flow2 = nn.BatchNorm2d(num_features = 3)
      self.flow_activation2 = nn.LeakyReLU()

      self.conv13 = nn.Conv2d(387, 64, 1, padding=0)
      self.batch_norm13 = nn.BatchNorm2d(num_features = 64)
      self.activation13 = nn.LeakyReLU()
      self.flow3 = nn.Conv2d(387, 3, 5, padding=2)
      self.batch_norm_flow3 = nn.BatchNorm2d(num_features = 3)
      self.flow_activation3 = nn.LeakyReLU()

      self.conv14 = nn.Conv2d(195, 3, 1, padding=0)
      self.batch_norm14 = nn.BatchNorm2d(num_features = 3)
      self.activation14 = nn.LeakyReLU()

      self.sigmoid = nn.Sigmoid()

      self.max_pool = nn.MaxPool2d(2)
      self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
      output1 = self.max_pool(self.activation1(self.batch_norm1(self.conv1(x))))
      output2 = self.max_pool(self.activation2(self.batch_norm2(self.conv2(output1))))
      output3 = self.max_pool(self.activation3(self.batch_norm3(self.conv3(output2))))
      output4 = self.activation4(self.batch_norm4(self.conv4(output3)))
      output5 = self.max_pool(self.activation5(self.batch_norm5(self.conv5(output4))))
      output6 = self.activation6(self.batch_norm6(self.conv6(output5)))
      output7 = self.max_pool(self.activation7(self.batch_norm7(self.conv7(output6))))
      output8 = self.activation8(self.batch_norm8(self.conv8(output7)))
      output9 = self.max_pool(self.activation9(self.batch_norm9(self.conv9(output8))))

      output10 = self.upsample(self.activation10(self.batch_norm10(self.conv10(output9))))

      output11 = self.upsample(self.activation11(self.batch_norm11(self.conv11(torch.cat((output10, output8), axis=1)))))
      flow1 = self.upsample(self.flow_activation1(self.batch_norm_flow1(self.flow1(torch.cat((output10, output8), axis=1)))))

      output12 = self.upsample(self.activation12(self.batch_norm12(self.conv12(torch.cat((output11, output6, flow1), axis=1)))))
      flow2 = self.upsample(self.flow_activation2(self.batch_norm_flow2(self.flow2(torch.cat((output11, output6, flow1), axis=1)))))

      output13 = self.upsample(self.activation13(self.batch_norm13(self.conv13(torch.cat((output12, output4, flow2), axis=1)))))
      flow3 = self.upsample(self.flow_activation3(self.batch_norm_flow3(self.flow3(torch.cat((output12, output4, flow2), axis=1)))))

      output = self.activation14(self.batch_norm14(self.conv14(torch.cat((output13, output2, flow3), axis=1))))
      #output = self.sigmoid(output)
      return output

class FlowNetS4(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(12, 192, 7, padding=3)
      self.batch_norm1 = nn.BatchNorm2d(num_features = 192)
      self.activation1 = nn.LeakyReLU()

      self.conv2 = nn.Conv2d(192, 384, 5, padding=2)
      self.batch_norm2 = nn.BatchNorm2d(num_features = 384)
      self.activation2 = nn.LeakyReLU()

      self.conv3 = nn.Conv2d(384, 768, 5, padding=2)
      self.batch_norm3 = nn.BatchNorm2d(num_features = 768)
      self.activation3 = nn.LeakyReLU()

      self.conv4 = nn.Conv2d(768, 768, 3, padding=1)
      self.batch_norm4 = nn.BatchNorm2d(num_features = 768)
      self.activation4 = nn.LeakyReLU()

      self.conv5 = nn.Conv2d(768, 1536, 3, padding=1)
      self.batch_norm5 = nn.BatchNorm2d(num_features = 1536)
      self.activation5 = nn.LeakyReLU()

      self.conv6 = nn.Conv2d(1536, 1536, 3, padding=1)
      self.batch_norm6 = nn.BatchNorm2d(num_features = 1536)
      self.activation6 = nn.LeakyReLU()

      self.conv7 = nn.Conv2d(1536, 1536, 3, padding=1)
      self.batch_norm7 = nn.BatchNorm2d(num_features = 1536)
      self.activation7 = nn.LeakyReLU()

      self.conv8 = nn.Conv2d(1536, 1536, 3, padding=1)
      self.batch_norm8 = nn.BatchNorm2d(num_features = 1536)
      self.activation8 = nn.LeakyReLU()

      self.conv9 = nn.Conv2d(1536, 3072, 3, padding=1)
      self.batch_norm9 = nn.BatchNorm2d(num_features = 3072)
      self.activation9 = nn.LeakyReLU()

      self.conv10 = nn.Conv2d(3072, 1536, 1, padding=0)
      self.batch_norm10 = nn.BatchNorm2d(num_features = 1536)
      self.activation10 = nn.LeakyReLU()

      self.conv11 = nn.Conv2d(3072, 768, 1, padding=0)
      self.batch_norm11 = nn.BatchNorm2d(num_features = 768)
      self.activation11 = nn.LeakyReLU()
      self.flow1 = nn.Conv2d(3072, 3, 5, padding=2)
      self.batch_norm_flow1 = nn.BatchNorm2d(num_features = 3)
      self.flow_activation1 = nn.LeakyReLU()


      self.conv12 = nn.Conv2d(2307, 384, 1, padding=0)
      self.batch_norm12 = nn.BatchNorm2d(num_features = 384)
      self.activation12 = nn.LeakyReLU()
      self.flow2 = nn.Conv2d(2307, 3, 5, padding=2)
      self.batch_norm_flow2 = nn.BatchNorm2d(num_features = 3)
      self.flow_activation2 = nn.LeakyReLU()

      self.conv13 = nn.Conv2d(1155, 192, 1, padding=0)
      self.batch_norm13 = nn.BatchNorm2d(num_features = 192)
      self.activation13 = nn.LeakyReLU()
      self.flow3 = nn.Conv2d(1155, 3, 5, padding=2)
      self.batch_norm_flow3 = nn.BatchNorm2d(num_features = 3)
      self.flow_activation3 = nn.LeakyReLU()

      self.conv14 = nn.Conv2d(579, 3, 1, padding=0)
      self.batch_norm14 = nn.BatchNorm2d(num_features = 3)
      self.activation14 = nn.LeakyReLU()

      self.sigmoid = nn.Sigmoid()

      self.max_pool = nn.MaxPool2d(2)
      self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
      output1 = self.max_pool(self.activation1(self.batch_norm1(self.conv1(x))))
      output2 = self.max_pool(self.activation2(self.batch_norm2(self.conv2(output1))))
      output3 = self.max_pool(self.activation3(self.batch_norm3(self.conv3(output2))))
      output4 = self.activation4(self.batch_norm4(self.conv4(output3)))
      output5 = self.max_pool(self.activation5(self.batch_norm5(self.conv5(output4))))
      output6 = self.activation6(self.batch_norm6(self.conv6(output5)))
      output7 = self.max_pool(self.activation7(self.batch_norm7(self.conv7(output6))))
      output8 = self.activation8(self.batch_norm8(self.conv8(output7)))
      output9 = self.max_pool(self.activation9(self.batch_norm9(self.conv9(output8))))

      output10 = self.upsample(self.activation10(self.batch_norm10(self.conv10(output9))))

      output11 = self.upsample(self.activation11(self.batch_norm11(self.conv11(torch.cat((output10, output8), axis=1)))))
      flow1 = self.upsample(self.flow_activation1(self.batch_norm_flow1(self.flow1(torch.cat((output10, output8), axis=1)))))

      output12 = self.upsample(self.activation12(self.batch_norm12(self.conv12(torch.cat((output11, output6, flow1), axis=1)))))
      flow2 = self.upsample(self.flow_activation2(self.batch_norm_flow2(self.flow2(torch.cat((output11, output6, flow1), axis=1)))))

      output13 = self.upsample(self.activation13(self.batch_norm13(self.conv13(torch.cat((output12, output4, flow2), axis=1)))))
      flow3 = self.upsample(self.flow_activation3(self.batch_norm_flow3(self.flow3(torch.cat((output12, output4, flow2), axis=1)))))

      output = self.activation14(self.batch_norm14(self.conv14(torch.cat((output13, output2, flow3), axis=1))))
      #output = self.sigmoid(output)
      return output

class FlowNetVector(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(14, 128, 7, padding=3)
    self.batch_norm1 = nn.BatchNorm2d(num_features = 128)
    self.activation1 = nn.LeakyReLU()

    self.conv2 = nn.Conv2d(128, 256, 5, padding=2)
    self.batch_norm2 = nn.BatchNorm2d(num_features = 256)
    self.activation2 = nn.LeakyReLU()

    self.conv3 = nn.Conv2d(256, 512, 5, padding=2)
    self.batch_norm3 = nn.BatchNorm2d(num_features = 512)
    self.activation3 = nn.LeakyReLU()

    self.conv4 = nn.Conv2d(512, 512, 3, padding=1)
    self.batch_norm4 = nn.BatchNorm2d(num_features = 512)
    self.activation4 = nn.LeakyReLU()

    self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
    self.batch_norm5 = nn.BatchNorm2d(num_features = 1024)
    self.activation5 = nn.LeakyReLU()

    self.conv6 = nn.Conv2d(1024, 1024, 3, padding=1)
    self.batch_norm6 = nn.BatchNorm2d(num_features = 1024)
    self.activation6 = nn.LeakyReLU()

    self.conv7 = nn.Conv2d(1024, 1024, 3, padding=1)
    self.batch_norm7 = nn.BatchNorm2d(num_features = 1024)
    self.activation7 = nn.LeakyReLU()

    self.conv8 = nn.Conv2d(1024, 1024, 3, padding=1)
    self.batch_norm8 = nn.BatchNorm2d(num_features = 1024)
    self.activation8 = nn.LeakyReLU()

    self.conv9 = nn.Conv2d(1024, 2048, 3, padding=1)
    self.batch_norm9 = nn.BatchNorm2d(num_features = 2048)
    self.activation9 = nn.LeakyReLU()

    self.conv10 = nn.Conv2d(2048, 1024, 1, padding=0)
    self.batch_norm10 = nn.BatchNorm2d(num_features = 1024)
    self.activation10 = nn.LeakyReLU()

    self.conv11 = nn.Conv2d(2048, 512, 1, padding=0)
    self.batch_norm11 = nn.BatchNorm2d(num_features = 512)
    self.activation11 = nn.LeakyReLU()
    self.flow1 = nn.Conv2d(2048, 3, 5, padding=2)
    self.batch_norm_flow1 = nn.BatchNorm2d(num_features = 3)
    self.flow_activation1 = nn.LeakyReLU()

    self.conv12 = nn.Conv2d(1539, 256, 1, padding=0)
    self.batch_norm12 = nn.BatchNorm2d(num_features = 256)
    self.activation12 = nn.LeakyReLU()
    self.flow2 = nn.Conv2d(1539, 3, 5, padding=2)
    self.batch_norm_flow2 = nn.BatchNorm2d(num_features = 3)
    self.flow_activation2 = nn.LeakyReLU()

    self.conv13 = nn.Conv2d(771, 128, 1, padding=0)
    self.batch_norm13 = nn.BatchNorm2d(num_features = 128)
    self.activation13 = nn.LeakyReLU()
    self.flow3 = nn.Conv2d(771, 3, 5, padding=2)
    self.batch_norm_flow3 = nn.BatchNorm2d(num_features = 3)
    self.flow_activation3 = nn.LeakyReLU()

    self.conv14 = nn.Conv2d(387, 3, 1, padding=0)
    self.batch_norm14 = nn.BatchNorm2d(num_features = 3)
    self.activation14 = nn.LeakyReLU()

    self.sigmoid = nn.Sigmoid()

    self.max_pool = nn.MaxPool2d(2)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

  def forward(self, x):
    output1 = self.max_pool(self.activation1(self.batch_norm1(self.conv1(x))))
    output2 = self.max_pool(self.activation2(self.batch_norm2(self.conv2(output1))))
    output3 = self.max_pool(self.activation3(self.batch_norm3(self.conv3(output2))))
    output4 = self.activation4(self.batch_norm4(self.conv4(output3)))
    output5 = self.max_pool(self.activation5(self.batch_norm5(self.conv5(output4))))
    output6 = self.activation6(self.batch_norm6(self.conv6(output5)))
    output7 = self.max_pool(self.activation7(self.batch_norm7(self.conv7(output6))))
    output8 = self.activation8(self.batch_norm8(self.conv8(output7)))
    output9 = self.max_pool(self.activation9(self.batch_norm9(self.conv9(output8))))

    output10 = self.upsample(self.activation10(self.batch_norm10(self.conv10(output9))))

    output11 = self.upsample(self.activation11(self.batch_norm11(self.conv11(torch.cat((output10, output8), axis=1)))))
    flow1 = self.upsample(self.flow_activation1(self.batch_norm_flow1(self.flow1(torch.cat((output10, output8), axis=1)))))

    output12 = self.upsample(self.activation12(self.batch_norm12(self.conv12(torch.cat((output11, output6, flow1), axis=1)))))
    flow2 = self.upsample(self.flow_activation2(self.batch_norm_flow2(self.flow2(torch.cat((output11, output6, flow1), axis=1)))))

    output13 = self.upsample(self.activation13(self.batch_norm13(self.conv13(torch.cat((output12, output4, flow2), axis=1)))))
    flow3 = self.upsample(self.flow_activation3(self.batch_norm_flow3(self.flow3(torch.cat((output12, output4, flow2), axis=1)))))

    output = self.activation14(self.batch_norm14(self.conv14(torch.cat((output13, output2, flow3), axis=1))))
    #output = self.sigmoid(output)
    return output
