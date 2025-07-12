import torch
import numpy as np
import os
from load_dataset import get_torcast_dataloader
from sklearn import metrics
from input_preprocessing import normalise_input

data_loader = get_torcast_dataloader("train", 2, 1)

item = next(iter(data_loader))
print(item)