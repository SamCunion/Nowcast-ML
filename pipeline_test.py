import torch
import numpy as np
import matplotlib.pyplot as plot
from tornet.display.display import plot_grid, get_cmap
import random
from load_dataset import get_torcast_dataloader
from input_preprocessing import normalise_input, interpolate_velocity_noise, reduce_to_dbz_threshold, detect_and_smooth_spikes, generate_feature_mask, remove_extreme_artefacts

def display_data(DBZ, VEL, RHOHV, EF_rating, title, mask=None):
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
    #show mask?
    if (mask != None):
        mask = mask.squeeze(0)
        ax1.imshow(mask, cmap="Purples", alpha=0.3, interpolation="nearest")
        ax2.imshow(mask, cmap="Purples", alpha=0.3, interpolation="nearest")
        ax3.imshow(mask, cmap="Purples", alpha=0.3, interpolation="nearest")

    plot.show()

data_loader = get_torcast_dataloader("train", 64, 10)
batch = next(iter(data_loader))
index = 30#random.randint(0, 63)
print(index)

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

#remove sidelobe artefacts
vel_data = detect_and_smooth_spikes(vel_data)
display_data(dbz_data, vel_data, rhohv_data, ef_number, "Removed velocity artefacts")

vel_data = remove_extreme_artefacts(vel_data)
display_data(dbz_data, vel_data, rhohv_data, ef_number, "Removed extreme velocity values")

#interpolate noisy velocity data
vel_data = interpolate_velocity_noise(dbz_data, vel_data)
display_data(dbz_data, vel_data, rhohv_data, ef_number, "Interpolated velocity")

#generate focus mask
mask = generate_feature_mask(dbz_data, vel_data, rhohv_data)[3]
display_data(dbz_data, vel_data, rhohv_data, ef_number, "Applied TDA", mask=mask)

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