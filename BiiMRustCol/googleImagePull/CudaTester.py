import os

import numpy as np

array = np.array([1,2,3,4,5])

import torch

print(torch.cuda.current_device())

print(torch.cuda.device_count())

print(torch.cuda.device_count())

print(torch.cuda.get_device_name(0))

print(torch.cuda.is_available())