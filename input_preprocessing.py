#TorCastML input preprocessing library
#TODO:
#Normalise inputs function
#identify velocity gates function
#shrink matrices to 50x50 around point function
#rotate matrice around true north instead of radar direction?
#recognise extreme values in velocity, and replace them with something more sensible?
#maybe take rhohv, take values that are above 1 and clamp to 1. maybe instead of nan being 0, make it 1?
import torch

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
            print("wut?")
            exit()
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
