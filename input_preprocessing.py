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

#generates a feature mask, which indicates the most important parts of the scan the model should pay attention to.
#uses a simplified version of the TDA to identify velocity couplets, builds bouding boxes around their centre point, and constructs a mask around them
#if too many couplets are found (could be on a boundary), or too few (0), instead masks the entire storm (not including low DBZ areas)
def generate_feature_mask(DBZ, VEL, RHOHV, DBZ_THRESHOLD=10):
    couplets = find_velocity_couplets(VEL)
    no_couplets = len(couplets)
    if (no_couplets == 0 or no_couplets > 8):
        #default to full scan, construct mask by dbz threshold
        MASK = DBZ > DBZ_THRESHOLD
        return DBZ, VEL, RHOHV, MASK
    
    #gets bounding box (X,Y,W,H) around centroids
    bb_list = centroid_to_bounding_box(couplets)
    #construct mask for centroids
    MASK = construct_bounding_box_mask(bb_list)
    #clip low dbz values from the bounding boxes
    MASK[DBZ.squeeze() < DBZ_THRESHOLD] = 0
    MASK = MASK.unsqueeze(0)
    return DBZ, VEL, RHOHV, MASK
    

#implements a simplified version of TDA, where we only care about opposing intense velocity values, in one dimension
def find_velocity_couplets(VEL, SHEAR_THRESHOLD=35.0):
    nand_vel = torch.nan_to_num(VEL.squeeze(), nan=0.0)
    #shifts in one direction, minus shift in other direction to get couplet shear
    shear = nand_vel[1:, :] - nand_vel[:-1, :]
    #mask for opposing direction
    mask = (nand_vel[1:, :] * nand_vel[:-1, :] < 0)
    strong_shear_mask = torch.abs(shear) > SHEAR_THRESHOLD
    #couplet detected where large difference between shear values, and direction
    velocity_couplet_mask = mask & strong_shear_mask
    rngs, azs = torch.where(velocity_couplet_mask)
    rngs = rngs + 1
    return list(zip(rngs.tolist(), azs.tolist()))

#constructs bounding boxes centred on the centroids of the identified velocity gates
def centroid_to_bounding_box(CENTROIDS, CROP_DIMS=(50, 50)):
    boxes = []
    for centroid in CENTROIDS:
        #RNG and AZ index of the center of rotation within the overall scan
        rng, az = centroid

        #calculate bounding box of crop
        rng_min = max(rng - CROP_DIMS[0] // 2, 0)
        rng_max = min(rng_min + CROP_DIMS[0], 120)
        az_min = max(az - CROP_DIMS[1] // 2, 0)
        az_max = min(az_min + CROP_DIMS[1], 240)

        boxes.append([rng_min, rng_max, az_min, az_max])
    return boxes

#using a list of bounding boxes, constructs a mask where 1 indicates a data value within a bounding box, and 0s are outside.
def construct_bounding_box_mask(bb_list):
    mask = torch.zeros(120, 240, dtype=torch.uint8) #hardcoded, will need change if input size varies
    for rngmin, rngmax, azmin, azmax in bb_list:
        #for every bounding box, set values between min,max for both dimensions to 1
        mask[rngmin:rngmax, azmin:azmax] = 1
    
    return mask

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

    #find velocity couplets
    DBZ, VEL, RHOHV, MASK = generate_feature_mask(DBZ, VEL, RHOHV)

    #normalise the inputs
    DBZ = normalise_input("DBZ", DBZ)
    VEL = normalise_input("VEL", VEL)
    RHOHV = normalise_input("RHOHV", RHOHV)

    if ((isinstance(DBZ, bool) and DBZ == False) or (isinstance(VEL, bool) and VEL == False) or (isinstance(RHOHV, bool) and RHOHV == False)):
        print("DBZ, VEL, or RHOHV matrix rejected, most likely either extremely low DBZ across the board, or VEL only in one direction")
        return False
    
    return DBZ, VEL, RHOHV, MASK