#tester tests a pretrained model with test data

from load_dataset import get_torcast_dataloader

data_loader = get_torcast_dataloader("test", 32, 10)

print("tester")