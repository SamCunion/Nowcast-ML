#uses the tornet dataset builder to load the dataset into what torcast expects

import tensorflow_datasets as tfds # need version >= 4.9.3
import tornet.data.tfds.tornet.tornet_dataset_builder # registers "tornet"
from tornet.data.torch.loader import make_torch_loader
import torch
from tornet.data.torch.loader import TFDSTornadoDataset

#train/test
YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022] #specifies all years to be used within dataset
ds = tfds.data_source('tornet')

#gets a dataloader that contains the relevant train/test (mode) samples
def get_torcast_dataloader(MODE, BATCH_SIZE, WORKERS):
    datasets = [TFDSTornadoDataset(ds["%s-%d" % (MODE, y)]) for y in YEARS] #gets the dataset for each year, of either the training or testing type
    dataset = torch.utils.data.ConcatDataset(datasets) #combines each year's dataset into one big pytorch dataset
    torch_dl = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=WORKERS) #constructs a pytorch dataloader from the dataset, with the supplied batch size and number of loader workers
    return torch_dl
