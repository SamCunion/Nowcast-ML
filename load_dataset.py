import tensorflow_datasets as tfds # need version >= 4.9.3
import tornet.data.tfds.tornet.tornet_dataset_builder # registers 'tornet'
from tornet.data.torch.loader import TFDSTornadoDataset

#train/test
MODE = "train"

years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
ds = tfds.data_source('tornet')

## Dataset, with preprocessing
transform = transforms.Compose([
    # transpose to [time,tile,az,rng]
    lambda d: permute_dims(d,(0,3,1,2)),
    # add coordinates tensor to data
    lambda d: add_coordinates(d,include_az=False,tilt_last=False,backend=torch), 
    # Remove time dimension
    lambda d: remove_time_dim(d)
])                                

datasets = [TFDSTornadoDataset(ds["%s-%d" % (MODE, y)], transform) for y in years]
dataset = torch.utils.data.ConcatDataset(datasets)
torch_dl = torch.utils.data.DataLoader( dataset, batch_size=16, num_workers=20)