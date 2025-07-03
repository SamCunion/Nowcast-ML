#tornet data builder, converts from slow files to array_record, much faster

import tensorflow_datasets as tfds
import TorCastML_dataset_builder

#dataset root file (contains "train" and "test" folders)
TORNET_ROOT = "/home/sam/Desktop/University Ubuntu Stuff/Project/dataset"

dl_config = tfds.download.DownloadConfig(manual_dir=TORNET_ROOT)
tfds.data_source("tornet", data_dir=TORNET_ROOT, builder_kwargs={"file_format": "array_record"}, download_and_prepare_kwargs={"download_config": dl_config})