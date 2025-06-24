#Ensures that all .nc files are accounted for in the dataset, according to the catalogue

import os
import sys
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

#set the dataset path
DATASET_PATH = os.getenv("DATASET_PATH")

print("Loading catalogue...")
catalogue = pd.read_csv(DATASET_PATH + "/catalog.csv")
print("Checking " + str(catalogue.size) + " items")
print("Checking integrity of dataset...")

success = 0

for index, item in catalogue.iterrows():
    filename = item["filename"]
    if os.path.exists(DATASET_PATH + "/" + filename):
        success += 1
    else:
        print("Unable to find file: " + filename)

print("Successfully found " + str(success) + " items")