import torch
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'test.pth.tar' #test.pth.tar
checkpoint = torch.load(model_path, map_location=str(device))
