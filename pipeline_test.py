import torch
import numpy as np
import matplotlib.pyplot as plot
import os
from tornet.display.display import plot_grid
import random
from load_dataset import get_torcast_dataloader
from sklearn import metrics
from input_preprocessing import normalise_input, interpolate_velocity_noise

def display_data(DBZ, VEL, RHOHV, EF_rating, title):
    figure = plot.figure(figsize=(12, 4))
    data = dict()
    data["DBZ"] = DBZ
    data["VEL"] = VEL
    data[RHOHV] = RHOHV
    plot_grid(data, fig=figure)
    figure.text(.5, .05, title + ", EF: " + str(EF_rating), ha="center")
    figure.show()

data_loader = get_torcast_dataloader("train", 2000, 1)


batch = next(iter(data_loader))
item = random.choice(batch)

dbz_data = item["DBZ"][...,0]
vel_data = item["VEL"][...,0]
rhohv_data = item["RHOHV"][...,0]
ef_number = item["ef_number"]

display_data(dbz_data, vel_data, rhohv_data, ef_number, "Raw, non-normalsied data")

#is one of the inputs filled with nans? red alert!
if (torch.isnan(dbz_data).all() or torch.isnan(vel_data).all() or torch.isnan(rhohv_data).all()):
    print("NANITEM!")
    print(item)

#interpolate noisy velocity data
vel_data = interpolate_velocity_noise(dbz_data, vel_data)

display_data(dbz_data, vel_data, rhohv_data, ef_number, "Interpolated velocity")

#normalise the inputs
norm_dbz = normalise_input("DBZ", dbz_data)
norm_vel = normalise_input("VEL", vel_data)
norm_rhohv = normalise_input("RHOHV", rhohv_data)

display_data(norm_dbz, norm_vel, norm_rhohv, ef_number, "Inputs normalised")

if (isinstance(norm_dbz, bool) and norm_dbz == False):
    nan_detected = True
    print("weird dbz matrix detected")
    print(item)
    exit()