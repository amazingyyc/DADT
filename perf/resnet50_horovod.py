# coding=utf-8

from __future__ import print_function, division

import torch, os, time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import horovod.torch as hvd

def train():
  hvd.init()

  # Data augmentation and normalization for training
  # Just normalization for validation
  data_transforms = {
      'train': transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  data_dir = 'hymenoptera_data'

  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'val']}

  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
              for x in ['train', 'val']}

  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  class_names = image_datasets['train'].classes

  device = torch.device("cuda:{}".format(hvd.local_rank()))
  print('device:', device)

  model_ft = models.resnet50(pretrained=False)
  num_ftrs = model_ft.fc.in_features

  # Here the size of each output sample is set to 2.
  # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
  model_ft.fc = nn.Linear(num_ftrs, 2)

  model_ft = model_ft.to(device)

  criterion = nn.CrossEntropyLoss()

  # Observe that all parameters are being optimized
  # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
  optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.002, momentum=0.9)

  # init distribute optimizer
  d_optimizer = hvd.DistributedOptimizer(optimizer=optimizer_ft, named_parameters=model_ft.named_parameters())
  hvd.broadcast_parameters(model_ft.state_dict(), root_rank=0)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

  total_cost_time = 0.0
  total_count = 0.0

  model_ft.train()
  for epoch in range(250):
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['train']:
      start_time = time.time()

      inputs = inputs.to(device)
      labels = labels.to(device)

      # zero the parameter gradients
      d_optimizer.zero_grad()

      outputs = model_ft(inputs)
      _, preds = torch.max(outputs, 1)

      loss = criterion(outputs, labels)

      loss.backward()
      d_optimizer.step()

      cost_time = int(round((time.time() - start_time) * 1000))

      total_cost_time += cost_time
      total_count += 1

      # statistics
      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)

      print('Rank:{}, cost time:{}, avg tiem:{} epoch:{}, loss:{}'.format(hvd.rank(), cost_time, total_cost_time/total_count, epoch, loss.item()))
      print('--------------------------------------------------------------------------')

    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_corrects.double() / dataset_sizes['train']

    print('Rank:{}, {} Loss: {:.4f} Acc: {:.4f}'.format(hvd.local_rank(), 'train', epoch_loss, epoch_acc))

if '__main__' == __name__:
  train()