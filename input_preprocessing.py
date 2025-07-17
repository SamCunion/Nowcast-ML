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

#removes sidelobe artefacts where velocity is set to -64.5 for some reason.
def remove_sidelobe_artefacts(VEL):
    new_vel = VEL.clone()
    new_vel[abs(new_vel) >= 64.0] = float("nan")
    return new_vel

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
            print("Completely void velocity")
            return False
        normed = torch.clamp(nansafe_matrix / abs_max, -1.0, 1.0)
    else:
        print("INVALID INPUT TYPE PASSED TO NORMALISE INPUT: " + type + ", EXITING PROCESS")
        exit()
    return normed

def preprocessing_pipeline(DBZ, VEL, RHOHV):

    #reject if filled with nans
    if (torch.isnan(DBZ).all() or torch.isnan(VEL).all() or torch.isnan(RHOHV).all()):
        print("DBZ, VEL or RHOHV matrix filled entirely with nan: rejected")
        return False

    #trehsold all 3 inputs yb DBZ
    DBZ, VEL, RHOHV = reduce_to_dbz_threshold(DBZ, VEL, RHOHV)

    #remove erroneous sidelobe values
    VEL = remove_sidelobe_artefacts(VEL)

    #gaussian smooth velocity data
    VEL = interpolate_velocity_noise(DBZ, VEL)

    #normalise the inputs
    DBZ = normalise_input("DBZ", DBZ)
    VEL = normalise_input("VEL", VEL)
    RHOHV = normalise_input("RHOHV", RHOHV)

    if (isinstance(DBZ, bool) and DBZ == False):
        print("DBZ, VEL, or RHOHV matrix rejected, most likely either extremely low DBZ across the board, or VEL only in one direction")
        return False
    
    return DBZ, VEL, RHOHV