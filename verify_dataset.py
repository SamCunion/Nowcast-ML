#Loops through the catalogue, and ensures that every sample is accounted for. Detects samples referenced by the catalogue that do not exist within the downloaded dataset.

import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

#set the dataset path
DATASET_PATH = os.getenv("DATASET_PATH")

print("Loading catalogue...")
catalogue = pd.read_csv(DATASET_PATH + "/catalog.csv") #reads catalogue into pandas dataframe
print("Checking " + str(catalogue.shape[0]) + " items")
print("Checking integrity of dataset...")

success = 0

for index, item in catalogue.iterrows():
    filename = item["filename"] #gets filename of sample
    if os.path.exists(DATASET_PATH + "/" + filename): #sample exists
        success += 1
    else: #sample does not exist
        print("Unable to find file: " + filename)

print("Successfully found " + str(success) + " items")