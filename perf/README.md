# How to test perf
## Download test data and unzip into folder hymenoptera_data
`
wget https://download.pytorch.org/tutorial/hymenoptera_data.zip && unzip hymenoptera_data.zip
`
## Run command for Horovod:
`
mpirun -np 4 python resnet101_horovod.py
`
## For DADT:
`
mpirun -np 4 python resnet101_dadt.py
`