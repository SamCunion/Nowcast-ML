#tester tests a pretrained model with test data

import torch
import numpy as np
import os
from load_dataset import get_torcast_dataloader
from sklearn import metrics
from input_preprocessing import normalise_input

#constants
MODEL_PATH = "./saved_models/" #saved models directory
TORNADO_PROBABILITY_THRESHOLD = 0.5 #tornado prediction probability threshold, above this value is considered an identification of a tornado

#entry
print("TorCast trainer module")
saved_model_list = os.listdir(MODEL_PATH)
print_str = "Select model to load:\n"
for i in range(0, len(saved_model_list)):
    print_str += "[" + str(i) + "] " + saved_model_list[i] + "\n"
print(print_str)
saved_model_id = int(input("Model ID: "))
print("begin testing?")
inp = input("Y/N: ")
if inp.lower() != "y":
    exit()

data_loader = get_torcast_dataloader("test", 32, 10)
model = torch.load(MODEL_PATH + saved_model_list[saved_model_id], weights_only=False)

#begin testing
print("Running TorCast Testing Module")
model.eval()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + "cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

tor_prob_predictions = []
tor_prob_truths = []
tor_strength_predictions = []
tor_strength_truths = []

no_batches = len(data_loader)
batches_trained = 0
with torch.no_grad():
    for batch in data_loader:
        batch_size = len(batch["label"])
        #only using the first radar tilt for now
        #PREPROCESSING

        #split batch before processing
        BATCH_DBZ = batch["DBZ"][...,0]
        BATCH_VEL = batch["VEL"][...,0]
        BATCH_RHOHV = batch["RHOHV"][...,0]
        SPLIT_DBZ = []
        SPLIT_VEL = []
        SPLIT_RHOHV = []

        for i in range(0, batch_size): #preprocess this item
            dbz_data = BATCH_DBZ[i] #individual dbz input
            vel_data = BATCH_VEL[i] #individual vel input
            rhohv_data = BATCH_RHOHV[i] #individual rhohv input

            #is one of the inputs filled with nans? red alert!
            if (torch.isnan(dbz_data).all() or torch.isnan(vel_data).all() or torch.isnan(rhohv_data).all()):
                print("BATCH CONTAINED ALL NAN DATA")
                continue

            norm_dbz = normalise_input("DBZ", dbz_data)
            norm_vel = normalise_input("VEL", vel_data)
            norm_rhohv = normalise_input("RHOHV", rhohv_data)

            if (isinstance(norm_dbz, bool) and norm_dbz == False):
                print("Weird dbz detected")
                continue

            SPLIT_DBZ.append(norm_dbz)
            SPLIT_VEL.append(norm_vel)
            SPLIT_RHOHV.append(norm_rhohv)
        
        #merge batch again
        DBZ = torch.stack(SPLIT_DBZ).to(DEVICE)
        VEL = torch.stack(SPLIT_VEL).to(DEVICE)
        RHOHV = torch.stack(SPLIT_RHOHV).to(DEVICE)
        label = batch["label"].cpu().numpy().squeeze().astype(int)
        ef_number = batch["ef_number"].cpu().numpy().squeeze().astype(int)

        prob, class_logits = model(DBZ, VEL, RHOHV)

        batch_tor_probs = torch.sigmoid(prob)
        batch_tor_predictions = (batch_tor_probs > TORNADO_PROBABILITY_THRESHOLD).int().view(-1).cpu().numpy()
        tor_prob_predictions.extend(batch_tor_predictions)
        tor_prob_truths.extend(label.astype(int))

        batch_strength_probs = torch.softmax(class_logits, dim=1)
        batch_strength_predictions = torch.argmax(batch_strength_probs, dim=1).cpu().numpy()
        tor_strength_predictions.extend(batch_strength_predictions)
        tor_strength_truths.extend(ef_number + 1) #+1 because we're converting -1 - 5 to 0 - 6 indexes


        #update visual
        batches_trained += 1
        print("Batch [" + str(batches_trained) + "/" + str(no_batches) + "] Tested")


#model evaluation
print("Calculating evaluation metrics...")

#tornado probability evaluation
accuracy = metrics.accuracy_score(tor_prob_predictions, tor_prob_truths)
precision = metrics.precision_score(tor_prob_predictions, tor_prob_truths)
recall = metrics.recall_score(tor_prob_predictions, tor_prob_truths)
f1 = metrics.f1_score(tor_prob_predictions, tor_prob_truths)
true_negatives, false_positives, false_negatives, true_positives = metrics.confusion_matrix(tor_prob_predictions, tor_prob_truths).ravel()

#tornado intensity evaluation
quad_kappa = metrics.cohen_kappa_score(tor_strength_truths, tor_strength_predictions, weights="quadratic")

#output
print("\n\n-----------------------------")
print("Results:")
print("-----------------------------")
print("Tornado Probability Head")
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))
print("\nTrue Positives: " + str(true_positives))
print("False Positives: " + str(false_positives))
print("True Negatives: " + str(true_negatives))
print("False Negatives: " + str(false_negatives))
print("-----------------------------")
print("Torando Intensity Head")
print("Quadratic Kappa: " + str(quad_kappa))
print("\n\n\nTesting completed!")