import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import cv2
import numpy as np
from loadimages import load_images
import random
from models import FlowNetS

sequences = load_images(1)[:10]
random.shuffle(sequences)

number_of_epochs = 100
model = FlowNetS()
loss_fn = nn.MSELoss()
# model.load_state_dict(torch.load('./etc.pt))
# model.cuda()
learning_rate = 1e-1
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

print('started training')
for epoch in range(number_of_epochs):
  training_loss = 0.0
  validation_loss = 0.0
  for data in sequences[:8]:
    out = model(data[0].unsqueeze(0))
    loss = loss_fn(out.squeeze(0), data[1])
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 6.0)
    training_loss += loss.item()
    optimizer.step()

  for data in sequences[8:]:
    out = model(data[0].unsqueeze(0))
    loss = loss_fn(out.squeeze(0), data[1])
    validation_loss += loss.item()
  print('Training loss: ' + str(training_loss) + ' ; Validation loss: ' + str(validation_loss))

