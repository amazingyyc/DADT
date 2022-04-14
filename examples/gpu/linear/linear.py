import torch
import torch.nn as nn
import numpy as np
import dadt.pytorch as dadt

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.block = torch.nn.Linear(1, 1)

  def forward(self, x):
    return self.block(x)

dadt.initialize(executor_type='nccl')

torch.cuda.set_device(dadt.local_rank())

model = Model().cuda()
criterion = nn.MSELoss().cuda()
optimizer = dadt.DistributedOptimizer(optimizer=torch.optim.SGD(model.parameters(), lr=0.001))

for i in range(10):
  x = torch.from_numpy(np.array([2], dtype=np.float32)).cuda()
  y = torch.from_numpy(np.array([3], dtype=np.float32)).cuda()

  pred = model(x)
  loss = criterion(pred, y)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  print('loss', loss.item())

# shut down background thread
dadt.shutdown()
