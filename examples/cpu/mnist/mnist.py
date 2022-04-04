# coding=utf-8

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import dadt.pytorch as dadt
import time

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output


def train(args, model, device, train_loader, distribte_optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    start_time = int(time.time() * 1000)

    data, target = data.to(device), target.to(device)

    distribte_optimizer.zero_grad()

    output = model(data)
    loss = F.nll_loss(output, target)

    loss.backward()
    distribte_optimizer.step()

    end_time = int(time.time() * 1000)

    if batch_idx % args.log_interval == 0:
      print(
          'Rank:{}, Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, cost time:{}'
          .format(dadt.rank(), epoch, batch_idx * len(data),
                  len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item(),
                  (end_time - start_time)))


def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target,
                              reduction='sum').item()  # sum up batch loss
      pred = output.argmax(
          dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print(
      '\nRank:{}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
      format(dadt.rank(), test_loss, correct, len(test_loader.dataset),
             100. * correct / len(test_loader.dataset)))


def log_grad_norm_sum(optimizer):
  tensors = []
  for param_group in optimizer.param_groups:
    for p in param_group['params']:
      if p.requires_grad:
        tensors.append(torch.norm(p.grad))

  t = torch.stack(tensors)

  print('Rank:', dadt.rank(), 'Grad norm sum:', torch.sum(t))


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size',
                      type=int,
                      default=64,
                      metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size',
                      type=int,
                      default=1000,
                      metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs',
                      type=int,
                      default=14,
                      metavar='N',
                      help='number of epochs to train (default: 14)')
  parser.add_argument('--lr',
                      type=float,
                      default=1.0,
                      metavar='LR',
                      help='learning rate (default: 1.0)')
  parser.add_argument('--gamma',
                      type=float,
                      default=0.7,
                      metavar='M',
                      help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--seed',
                      type=int,
                      default=1,
                      metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument(
      '--log-interval',
      type=int,
      default=10,
      metavar='N',
      help='how many batches to wait before logging training status')
  parser.add_argument('--save-model',
                      action='store_true',
                      default=False,
                      help='For Saving the current Model')
  args = parser.parse_args()

  # initialize dadt
  dadt.initialize()

  torch.manual_seed(args.seed)

  # get device by rank
  device = torch.device('cpu')

  kwargs = {'batch_size': args.batch_size}
  # kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True},)

  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))])

  dataset1 = datasets.MNIST('./data{}'.format(dadt.local_rank()),
                            train=True,
                            download=True,
                            transform=transform)
  dataset2 = datasets.MNIST('./data{}'.format(dadt.local_rank()),
                            train=False,
                            transform=transform)

  train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

  model = Net().to(device)
  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
  distribte_optimizer = dadt.DistributedOptimizer(optimizer=optimizer)

  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

  for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, distribte_optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

  if args.save_model and 0 == dadt.rank():
    torch.save(model.state_dict(), "mnist_cnn.pt")

  # shut down background thread
  dadt.shutdown()


if __name__ == '__main__':
  main()
