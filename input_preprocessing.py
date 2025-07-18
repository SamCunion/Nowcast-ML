#TorCastML input preprocessing library
#TODO:
#
#identify velocity gates function
#shrink matrices to 50x50 around point function
#rotate matrice around true north instead of radar direction?

import torch
import scipy
import numpy as np

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

#removes artefacts where the absolute value of a datapoint is more than their 4 nearest neighbours combined. If so, replaces it with the average of its 4 nearest neighbours
def detect_and_smooth_spikes(VEL):
    new_vel = VEL.clone().squeeze()
    nand = torch.nan_to_num(new_vel, nan=0.0)
    height, width = new_vel.shape
    
    #pad so edges work
    padded = torch.nn.functional.pad(nand.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate").squeeze()

    #non-absolute neighbour values
    up = padded[0:height, 1:width + 1]
    down = padded[2:height + 2, 1:width + 1]
    left = padded[1:height + 1, 0:width]
    right = padded[1:height + 1, 2: width + 2]

    #absolute values for the base matrix, and the sum of each datapoints 4 nearest neighbours
    input_abs = torch.abs(nand)
    neighbour_sums = torch.abs(up) + torch.abs(down) + torch.abs(left) + torch.abs(right)

    #these values hold the average value of each datapoints nearest neighbours
    true_neighbour_average = (up + down + left + right) / 4

    #mask for each datapoint that has higher abs value than neighbours abs values 
    spike_mask = input_abs > neighbour_sums

    #replaces the found datapoints with their nearest neighbour computed average
    new_vel[spike_mask] = true_neighbour_average[spike_mask]

    return new_vel.unsqueeze(0)

#interpolates velocity data to fill in holes where the corresponding DBZ is greater than a value
def interpolate_velocity_noise(DBZ, VEL, SIGMA=2.0):
    device = VEL.device
    #mask where velocity is NAN and DBZ exists
    mask = torch.isnan(VEL) & ~torch.isnan(DBZ)
    #nan removed velocity for smoothing purposes
    filled_vel = torch.nan_to_num(VEL, nan=0.0)

    #gaussian kernel, for convolution in 2 dimensions
    kernel_size = int(6 * SIGMA + 1)
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gaussian = torch.exp(-0.5 * (coords / SIGMA)**2)
    kernel = (gaussian[:, None] @ gaussian[None, :])
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    #performs convolution on the velocity to smooth
    smoothed_velocity = torch.nn.functional.conv2d(filled_vel, kernel, padding=kernel_size // 2)

    out = VEL.clone()
    out[0][mask[0]] = smoothed_velocity.squeeze(0)[mask[0]]
    return out

#Normalises matrix values between 0,1 for DBZ, RHOHV, between -1,1 for VEL. also converts NAN to 0
def normalise_input(type, matrix):
    if (type == "DBZ"): #0,1
        #convert nans
        nansafe_matrix = torch.nan_to_num(matrix, nan=0.0)
        #simple cast all values from max,min to 0,1
        min_val = torch.min(nansafe_matrix)
        max_val = torch.max(nansafe_matrix)
        if (min_val == max_val):
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
            return False
        normed = torch.clamp(nansafe_matrix / abs_max, -1.0, 1.0)
    else:
        print("INVALID INPUT TYPE PASSED TO NORMALISE INPUT: " + type + ", EXITING PROCESS")
        exit()
    return normed

def attempt_shrink_by_tda(DBZ, VEL, RHOHV):
    couplets = find_velocity_couplets(VEL)
    print(len(couplets))


def find_velocity_couplets(VEL, SHEAR_THRESHOLD=35.0):
    nand_vel = torch.nan_to_num(VEL.squeeze(), nan=0.0)
    #shifts in one direction, minus shift in other direction to get couplet shear
    shear = nand_vel[:, 1:] - nand_vel[:, :-1]
    #mask for opposing direction
    mask = (nand_vel[:, 1:] * nand_vel[:, :-1] < 0)
    strong_shear_mask = torch.abs(shear) > SHEAR_THRESHOLD
    #couplet detected where large difference between shear values, and direction
    velocity_couplet_mask = mask & strong_shear_mask
    azs, rngs = torch.where(velocity_couplet_mask)
    rngs = rngs + 1
    return list(zip(azs.tolist(), rngs.tolist()))

def preprocessing_pipeline(DBZ, VEL, RHOHV):

    #reject if filled with nans
    if (torch.isnan(DBZ).all() or torch.isnan(VEL).all() or torch.isnan(RHOHV).all()):
        print("DBZ, VEL or RHOHV matrix filled entirely with nan: rejected")
        return False

    #trehsold all 3 inputs yb DBZ
    DBZ, VEL, RHOHV = reduce_to_dbz_threshold(DBZ, VEL, RHOHV)

    #remove erroneous sidelobe values
    VEL = detect_and_smooth_spikes(VEL)

    #gaussian smooth velocity data
    VEL = interpolate_velocity_noise(DBZ, VEL)

    #find velocity couplets, and shrink if needs be
    attempt_shrink_by_tda(DBZ, VEL, RHOHV)

    #normalise the inputs
    DBZ = normalise_input("DBZ", DBZ)
    VEL = normalise_input("VEL", VEL)
    RHOHV = normalise_input("RHOHV", RHOHV)

    if ((isinstance(DBZ, bool) and DBZ == False) or (isinstance(VEL, bool) and VEL == False) or (isinstance(RHOHV, bool) and RHOHV == False)):
        print("DBZ, VEL, or RHOHV matrix rejected, most likely either extremely low DBZ across the board, or VEL only in one direction")
        return False
    
    return DBZ, VEL, RHOHV