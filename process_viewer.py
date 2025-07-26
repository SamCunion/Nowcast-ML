#select a netcdf file, takes you through the pipeline and shows the output with a given model.
#if no netcdf selected (None), chooses a random item
import torch
#import scipy
import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from input_preprocessing import reduce_to_dbz_threshold, detect_and_smooth_spikes, remove_extreme_artefacts, interpolate_velocity_noise, generate_feature_mask, normalise_input
from tornet.data.loader import read_file
from tornet.display.display import get_cmap
from inference import Query_Model
from dotenv import load_dotenv
load_dotenv()


#Hyperparams
DATASET_PATH = os.getenv("DATASET_PATH")
MODEL_PATH = "./saved_models/TorCastML-10e_new_training.pt"
SAMPLE = "test/2015/TOR_151223_230807_KNQA_610239_P2.nc"

#==============================================================================================

def plot_images(DBZ, VEL, RHOHV, MASK=None, CAM=None):
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    fields = [("DBZ", DBZ.squeeze(0)), ("VEL", VEL.squeeze(0)), ("RHOHV", RHOHV.squeeze(0))]
    for i, (title, field) in enumerate(fields):
        axes[i].imshow(field, cmap=get_cmap(title.lower())[0])
        axes[i].set_title(title)
        axes[i].axis("off")

        if (MASK != None):
            axes[i].imshow(MASK.squeeze(0), cmap="Purples", alpha=0.3, interpolation="nearest")
        
        if (CAM != None):
            axes[i].imshow(CAM, cmap="jet", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    catalogue = pd.read_csv(DATASET_PATH + "/catalog.csv")
    #model = torch.load(MODEL_PATH)

    if (SAMPLE == None): #get random sample for viewing
        random_item = catalogue.sample(n=1)
        filename = random_item["filename"].values[0]
        if os.path.exists(DATASET_PATH + "/" + filename):
            SAMPLE = filename
        else:
            print("Unable to find file: " + filename)
            exit()
    
    cdf_file = read_file(DATASET_PATH + "/" + SAMPLE)
    catalogue_item = catalogue.loc[catalogue["filename"] == SAMPLE]
    catalogue_item = catalogue_item.to_numpy()[0]
    ef_number = str(catalogue_item[8])
    sample_type = catalogue_item[9]
    timestamp = catalogue_item[1]
    radar_id = catalogue_item[7]

    DBZ = cdf_file["DBZ"][0][:, :, 1]
    VEL = cdf_file["VEL"][0][:, :, 1]
    RHOHV = cdf_file["RHOHV"][0][:, :, 1]
    #convert inputs
    DBZ = torch.from_numpy(DBZ).unsqueeze(0)
    VEL = torch.from_numpy(VEL).unsqueeze(0)
    RHOHV = torch.from_numpy(RHOHV).unsqueeze(0)
    MASK = None
    CAM = None
    #shape: (120, 240)


    #PREPROCESSING PIPELINE

    #plot raw inputs
    title = "Raw inputs"
    plot_images(DBZ, VEL, RHOHV)

    #is one of the inputs filled with nans? red alert!
    if (torch.isnan(DBZ).all() or torch.isnan(VEL).all() or torch.isnan(RHOHV).all()):
        print("Sample filled with NaNs, rejected!")

    title = "Reduced to DBZ threshold"
    DBZ, VEL, RHOHV = reduce_to_dbz_threshold(DBZ, VEL, RHOHV)
    plot_images(DBZ, VEL, RHOHV)

    title = "Smoothed velocity spikes"
    VEL = detect_and_smooth_spikes(VEL)
    plot_images(DBZ, VEL, RHOHV)

    title = "Removed extreme artefacts"
    VEL = remove_extreme_artefacts(VEL)
    plot_images(DBZ, VEL, RHOHV)

    title = "Gaussian smoothed velocity in noisy areas"
    VEL = interpolate_velocity_noise(DBZ, VEL)
    plot_images(DBZ, VEL, RHOHV)

    title = "Constructed mask around areas of interest"
    DBZ, VEL, RHOHV, MASK = generate_feature_mask(DBZ, VEL, RHOHV)
    plot_images(DBZ, VEL, RHOHV, MASK=MASK)
    
    title = "Normalised the data"
    DBZ = normalise_input("DBZ", DBZ)
    VEL = normalise_input("VEL", VEL)
    RHOHV = normalise_input("RHOHV", RHOHV)
    plot_images(DBZ, VEL, RHOHV, MASK=MASK)

    if ((isinstance(DBZ, bool) and DBZ == False) or (isinstance(VEL, bool) and VEL == False) or (isinstance(RHOHV, bool) and RHOHV == False)):
        print("DBZ, VEL, or RHOHV matrix rejected, most likely either extremely low DBZ across the board, or VEL only in one direction")
        exit()

    #build inputs
    title = "Model output and GRAD-CAM attention heatmap"
    if (len(model.gradcam_targets) > 1): #combined head
        stack = torch.cat([DBZ, VEL, RHOHV, MASK], dim=0)
        TOR_PROB, CLASS_PROBS, CAM = Query_Model(model, stack=stack, with_grad=True)
    else:
        TOR_PROB, CLASS_PROBS, CAM = Query_Model(model, DBZ=DBZ, VEL=VEL, RHOHV=RHOHV, with_grad=True)
    plot_images(DBZ, VEL, RHOHV, CAM=CAM)
