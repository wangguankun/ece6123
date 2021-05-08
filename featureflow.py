import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

gradient_x = [
  [1, 0, -1],
  [2, 0, -2],
  [1, 0, -1]
]

gradient_y = [
  [1, 2, 1],
  [0, 0, 0],
  [-1, -2, -1]
]

padded_image_1 = np.pad(img_1)
padded_image_2 = np.pad(img_2)

gradient_x_img_1 = np.zeros((img_1.shape[1], img_1.shape[0]))
gradient_y_img_1 = np.zeros((img_1.shape[1], img_1.shape[0]))
gradient_x_img_2 = np.zeros((img_2.shape[1], img_2.shape[0]))
gradient_y_img_2 = np.zeros((img_2.shape[1], img_2.shape[0]))

img_1_edges = np.zeros((img_1.shape[1], img_1.shape[0]))
img_2_edges = np.zeros((img_2.shape[1], img_2.shape[0]))

for i in range(1, len(padded_image_1) - 1):
  for j in range(1, len(padded_image_1) - 1):
    gradient_x_img_1[i][j] = (gradient_x[i - 1][j - 1] * padded_image_1[i - 1][j - 1] + gradient_x[i - 1][j] * padded_image_1[i - 1][j] + gradient_x[i - 1][j + 1] * padded_image_1[i - 1][j + 1]
     + gradient_x[i][j - 1] * padded_image_1[i][j - 1] + gradient_x[i][j] * padded_image_1[i][j] + gradient_x[i][j + 1] * padded_image_1[i][j + 1]
     + gradient_x[i + 1][j - 1] * padded_image_1[i + 1][j - 1] + gradient_x[i + 1][j] * padded_image_1[i + 1][j] + gradient_x[i + 1][j + 1] * padded_image_1[i + 1][j + 1])
    gradient_y_img_1[i][j] = (gradient_x[i - 1][j - 1] * padded_image_1[i - 1][j - 1] + gradient_x[i - 1][j] * padded_image_1[i - 1][j] + gradient_x[i - 1][j + 1] * padded_image_1[i - 1][j + 1]
     + gradient_x[i][j - 1] * padded_image_1[i][j - 1] + gradient_x[i][j] * padded_image_1[i][j] + gradient_x[i][j + 1]  * padded_image_1[i][j + 1]
     + gradient_x[i + 1][j - 1] * padded_image_1[i + 1][j - 1] + gradient_x[i + 1][j] * padded_image_1[i + 1][j] + gradient_x[i + 1][j + 1] * padded_image_1[i + 1][j + 1])
    gradient_x_img_2[i][j] = (gradient_x[i - 1][j - 1] * padded_image_2[i - 1][j - 1] + gradient_x[i - 1][j] * padded_image_2[i - 1][j] + gradient_x[i - 1][j + 1] * padded_image_2[i - 1][j + 1]
     + gradient_x[i][j - 1] * padded_image_2[i][j - 1] + gradient_x[i][j] * padded_image_2[i][j] + gradient_x[i][j + 1] * padded_image_2[i][j + 1]
     + gradient_x[i + 1][j - 1] * padded_image_2[i + 1][j - 1] + gradient_x[i + 1][j] * padded_image_2[i + 1][j] + gradient_x[i + 1][j + 1] * padded_image_2[i + 1][j + 1])
    gradient_y_img_2[i][j] = (gradient_x[i - 1][j - 1] * padded_image_2[i - 1][j - 1] + gradient_x[i - 1][j] * padded_image_2[i - 1][j] + gradient_x[i - 1][j + 1] * padded_image_2[i - 1][j + 1]
     + gradient_x[i][j - 1] * padded_image_2[i][j - 1] + gradient_x[i][j] * padded_image_2[i][j] + gradient_x[i][j + 1]  * padded_image_2[i][j + 1]
     + gradient_x[i + 1][j - 1] * padded_image_2[i + 1][j - 1] + gradient_x[i + 1][j] * padded_image_2[i + 1][j] + gradient_x[i + 1][j + 1] * padded_image_2[i + 1][j + 1])

    img_1_edges[i][j] = (gradient_x_img_1[i][j] ** 2 + gradient_y_img_1[i][j] ** 2) ** (0.5)
    img_2_edges[i][j] = (gradient_x_img_2[i][j] ** 2 + gradient_y_img_2[i][j] ** 2) ** (0.5)

