import torch
import numpy as np
import matplotlib.pyplot as plot
from tornet.display.display import plot_grid, get_cmap
import random
from load_dataset import get_torcast_dataloader
from input_preprocessing import normalise_input, interpolate_velocity_noise, reduce_to_dbz_threshold

def display_data(DBZ, VEL, RHOHV, EF_rating, title):
    figure = plot.figure(figsize=(12, 4))
    cmapv, normv = get_cmap("vel")
    cmapd, normd = get_cmap("dbz")
    cmapc, normc = get_cmap("rhohv")
    ax1 = figure.add_subplot(1, 3, 1)
    ax2 = figure.add_subplot(1, 3, 2)
    ax3 = figure.add_subplot(1, 3, 3)
    ax1.imshow(DBZ[0], origin="upper", cmap=cmapd)
    ax2.imshow(VEL[0], origin="upper", cmap=cmapv)
    ax3.imshow(RHOHV[0], origin="upper", cmap=cmapc)
    ax1.tick_params(axis="both", which="both", length=0)
    ax2.tick_params(axis="both", which="both", length=0)
    ax3.tick_params(axis="both", which="both", length=0)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    figure.text(.5, .05, title + ", EF: " + str(EF_rating), ha="center")
    plot.show()

data_loader = get_torcast_dataloader("train", 64, 10)
batch = next(iter(data_loader))
index = random.randint(0, 63)

dbz_data = batch["DBZ"][...,0][index]
vel_data = batch["VEL"][...,0][index]
rhohv_data = batch["RHOHV"][...,0][index]
ef_number = batch["ef_number"][index][0].item()

display_data(dbz_data, vel_data, rhohv_data, ef_number, "Raw, non-normalsied data")

#is one of the inputs filled with nans? red alert!
if (torch.isnan(dbz_data).all() or torch.isnan(vel_data).all() or torch.isnan(rhohv_data).all()):
    print("NANITEM!")
    print(batch["ef_number"][index])

#discard data where DBZ is less than a threshold (default 20)
dbz_data, vel_data, rhohv_data = reduce_to_dbz_threshold(dbz_data, vel_data, rhohv_data)
display_data(dbz_data, vel_data, rhohv_data, ef_number, "Reduced to DBZ threshold")

#interpolate noisy velocity data
vel_data = interpolate_velocity_noise(vel_data)

display_data(dbz_data, vel_data, rhohv_data, ef_number, "Interpolated velocity")


#normalise the inputs
dbz_data = normalise_input("DBZ", dbz_data)
vel_data = normalise_input("VEL", vel_data)
rhohv_data = normalise_input("RHOHV", rhohv_data)

display_data(dbz_data, vel_data, rhohv_data, ef_number, "Inputs normalised")

if (isinstance(dbz_data, bool) and dbz_data == False):
    nan_detected = True
    print("weird dbz matrix detected")
    print(batch["ef_number"][index])
    exit()