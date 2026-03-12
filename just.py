import torch
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2, 3"
print('--------------------------------')
print("Python version:", sys.version)
print('cuda_version:   ',torch.version.cuda)
print('pytorch_version:',torch.__version__)
#print('device_name',torch.cuda.get_device_name())
print('device_count:   ',torch.cuda.device_count())
#print('cuda.is_available:',torch.cuda.is_available())
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU device {i}: {device_name}")
else:
    print("No GPU devices available.")

print('cudnn_version:  ',torch.backends.cudnn.version())
print('--------------------------------')