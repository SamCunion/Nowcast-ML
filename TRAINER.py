#trainer trains the model with training data

from load_dataset import get_torcast_dataloader

data_loader = get_torcast_dataloader("train", 32, 20)

print("trainer")