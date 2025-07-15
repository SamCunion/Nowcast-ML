#TorCastML input preprocessing library
#TODO:
#
#identify velocity gates function
#shrink matrices to 50x50 around point function
#rotate matrice around true north instead of radar direction?
#recognise extreme values in velocity, and replace them with something more sensible?

import torch
import scipy
import numpy as np

#interpolates velocity data to fill in holes where the corresponding DBZ is greater than a value
def interpolate_velocity_noise(DBZ, VEL, DBZ_THRESHOLD=20, SIGMA=2.0):
    device = VEL.device
    #mask where velocity is NAN and DBZ > threshold
    mask = torch.isnan(VEL) & (DBZ > DBZ_THRESHOLD)
    #nan removed velocity for smoothing purposes
    filled_vel = torch.nan_to_num(VEL, nan=0.0)
    #weights to remove data where reflectivity is lower than the threshold from consideration
    weighted_reflectivity = (DBZ > DBZ_THRESHOLD).float()

    #gaussian kernel, for convolution in 2 dimensions
    kernel_size = int(6 * SIGMA + 1)
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gaussian = torch.exp(-0.5 * (coords / SIGMA)**2)
    kernel = (gaussian[:, None] @ gaussian[None, :])
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    #gets masks velocity only where reflectivity is sufficcient
    velocity = (filled_vel * weighted_reflectivity).unsqueeze(0)
    dbz = weighted_reflectivity.unsqueeze(0)

    #performs convolution on the velocity and the dbz weights
    smoothed_velocity = torch.nn.functional.conv2d(velocity, kernel, padding=kernel_size // 2)
    smoothed_dbz = torch.nn.functional.conv2d(dbz, kernel, padding=kernel_size // 2)
    #higher DBZ has more influence on smoothed velocity
    combined = smoothed_velocity.squeeze() / (smoothed_dbz.squeeze() + 1e-6)

    out = VEL.clone()
    out[0][mask[0]] = combined[mask[0]]
    return out

#removes all data where dbz is less than a threshold.
def reduce_to_dbz_threshold(DBZ, VEL, RHOHV, DBZ_THRESHOLD=20):
    #get new tensors
    new_dbz = DBZ.clone()
    new_vel = VEL.clone()
    new_rho = RHOHV.clone()
    #calcualte mask, 1s where threshold is not met
    mask = new_dbz < DBZ_THRESHOLD
    #set values where threshold is not met to nan
    new_dbz[mask] = float("nan")
    new_vel[mask] = float("nan")
    new_rho[mask] = float("nan")
    #return new tensors
    return new_dbz, new_vel, new_rho

#Normalises matrix values between 0,1 for DBZ, RHOHV, between -1,1 for VEL. also converts NAN to 0
def normalise_input(type, matrix):
    if (type == "DBZ"): #0,1
        #convert nans
        nansafe_matrix = torch.nan_to_num(matrix, nan=0.0)
        #remove negative values
        nansafe_matrix = torch.clamp(nansafe_matrix, min=0.0)
        #simple cast all values from max,min to 0,1
        min_val = torch.min(nansafe_matrix)
        max_val = torch.max(nansafe_matrix)
        if (min_val == max_val):
            print(min_val)
            print(max_val)
            print(matrix)
            return False
        normed = (nansafe_matrix - min_val) / (max_val - min_val)
    elif (type == "RHOHV"):
        #convert nans
        nansafe_matrix = torch.nan_to_num(matrix, nan=1.0)
        #clamp values between 0 and 1
        normed = torch.clamp(nansafe_matrix, 0.0, 1.0)
    elif (type == "VEL"): #-1,1
        #convert nans
        nansafe_matrix = torch.nan_to_num(matrix, nan=0.0)
        #get the highest wind speed (in either direction), which will be represented by -1.0 and 1.0. all values then fall between these extremes
        abs_max = torch.max(torch.abs(nansafe_matrix))
        if (abs_max == 0):
            print("AHA THERE WE GO")
            exit()
        normed = torch.clamp(nansafe_matrix / abs_max, -1.0, 1.0)
    else:
        print("INVALID INPUT TYPE PASSED TO NORMALISE INPUT: " + type + ", EXITING PROCESS")
        exit()
    return normed


#testing
if __name__ == "__main__":
    from load_dataset import get_torcast_dataloader
    dl = get_torcast_dataloader("test", 1, 1)
    for batch in dl:
        print(batch)
        batch_size = len(batch)
        #split batch before processing
        BATCH_DBZ = batch["DBZ"][...,0]
        BATCH_VEL = batch["VEL"][...,0]
        BATCH_RHOHV = batch["RHOHV"][...,0]

        dbz_data = BATCH_DBZ[0] #individual dbz input
        vel_data = BATCH_VEL[0] #individual vel input
        rhohv_data = BATCH_RHOHV[0] #individual rhohv input

        norm_dbz = normalise_input("DBZ", dbz_data)
        norm_vel = normalise_input("VEL", vel_data)
        norm_rhohv = normalise_input("RHOHV", rhohv_data)

        print("DBZ normalised:")
        print(norm_dbz)
        print("VEL normalsied:")
        print(norm_vel)
        print("RHOHV Normalised:")
        print(norm_rhohv)
            
        break
