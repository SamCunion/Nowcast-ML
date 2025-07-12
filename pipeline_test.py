import torch
import numpy as np
import os
from load_dataset import get_torcast_dataloader
from sklearn import metrics
from input_preprocessing import normalise_input

data_loader = get_torcast_dataloader("train", 2, 1)

item = next(iter(data_loader))
print("RAW BATCH:")
print(item)

 #split batch before processing
BATCH_DBZ = item["DBZ"][...,0]
BATCH_VEL = item["VEL"][...,0]
BATCH_RHOHV = item["RHOHV"][...,0]
for i in range(0, 2): #preprocess this item
    dbz_data = BATCH_DBZ[i] #individual dbz input
    vel_data = BATCH_VEL[i] #individual vel input
    rhohv_data = BATCH_RHOHV[i] #individual rhohv input

    norm_dbz = normalise_input("DBZ", dbz_data)
    norm_vel = normalise_input("VEL", vel_data)
    norm_rhohv = normalise_input("RHOHV", rhohv_data)
    print(norm_dbz)
    print("Raw input stats:", dbz_data.min(), dbz_data.max(), dbz_data.mean())
    print("Normed input stats:", norm_dbz.min(), norm_dbz.max(), norm_dbz.mean())

    print(norm_vel)
    print("Raw input stats:", vel_data.min(), vel_data.max(), vel_data.mean())
    print("Normed input stats:", norm_vel.min(), norm_vel.max(), norm_vel.mean())

    print(norm_rhohv)
    print("Raw input stats:", rhohv_data.min(), rhohv_data.max(), rhohv_data.mean())
    print("Normed input stats:", norm_rhohv.min(), norm_rhohv.max(), norm_rhohv.mean())
        
label = item["label"].cpu().numpy()
ef_number = item["ef_number"].cpu().numpy()

print("label:")
print(label)
print("EF: ")
print(ef_number)