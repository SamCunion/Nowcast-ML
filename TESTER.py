#tester tests a pretrained model with test data

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from load_dataset import get_torcast_dataloader
from sklearn import metrics
from collections import Counter
from input_preprocessing import preprocessing_pipeline

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
sample_categories = []
percentages = []

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
        BATCH_LABEL = batch["label"].squeeze().float()
        BATCH_EF = batch["ef_number"].squeeze().long()
        BATCH_CATEGORY = batch["category"].squeeze().long()
        SPLIT_STACK = []
        SPLIT_DBZ = []
        SPLIT_VEL = []
        SPLIT_RHOHV = []
        SPLIT_LABEL = []
        SPLIT_CATEGORY = []
        SPLIT_EF = []

        for i in range(0, batch_size): #preprocess this item
            dbz_data = BATCH_DBZ[i] #individual dbz input
            vel_data = BATCH_VEL[i] #individual vel input
            rhohv_data = BATCH_RHOHV[i] #individual rhohv input
            label_data = BATCH_LABEL[i] #individual label
            ef_data = BATCH_EF[i] #individual ef number
            cat_data = BATCH_CATEGORY[i] #individual category label

            matrices = preprocessing_pipeline(dbz_data, vel_data, rhohv_data)
            if (isinstance(matrices, bool) and matrices == False):
                #invalid, just skip this item in the batch
                continue

            #FOR v1
            #combine proessed matrices with mask
            MASKED_DBZ = torch.cat([matrices[0], matrices[3]], dim=0)
            MASKED_VEL = torch.cat([matrices[1], matrices[3]], dim=0)
            MASKED_RHOHV = torch.cat([matrices[2], matrices[3]], dim=0)
            SPLIT_DBZ.append(MASKED_DBZ)
            SPLIT_VEL.append(MASKED_VEL)
            SPLIT_RHOHV.append(MASKED_RHOHV)

            ##FOR v2:
            #combine scans with the mask to create the stack (DBZ, VEL, RHOHV, MASK)
            #STACK = torch.cat([matrices[0], matrices[1], matrices[2], matrices[3]], dim=0)
            #SPLIT_STACK.append(STACK)


            SPLIT_LABEL.append(label_data)
            SPLIT_EF.append(ef_data)
            SPLIT_CATEGORY.append(cat_data)
        
        
        #merge batch again
        #FOR v1:
        DBZ = torch.stack(SPLIT_DBZ, dim=0).to(DEVICE)
        VEL = torch.stack(SPLIT_VEL, dim=0).to(DEVICE)
        RHOHV = torch.stack(SPLIT_RHOHV, dim=0).to(DEVICE)

        ##FOR v2:
        #INPUT_STACK = torch.stack(SPLIT_STACK).to(DEVICE)

        labels = [val.item() for val in SPLIT_LABEL]
        ef_numbers = [int(val.item()) + 1 for val in SPLIT_EF]
        categories = [int(val.item()) for val in SPLIT_CATEGORY]

        prob, class_logits = model(DBZ, VEL, RHOHV)

        batch_tor_probs = torch.sigmoid(prob)
        batch_tor_predictions = (batch_tor_probs > TORNADO_PROBABILITY_THRESHOLD).int().view(-1).cpu().numpy()
        tor_prob_predictions.extend(batch_tor_predictions)
        tor_prob_truths.extend(labels)
        sample_categories.extend(categories)
        percentages.extend(batch_tor_probs)

        batch_strength_probs = torch.softmax(class_logits, dim=1)
        batch_strength_predictions = torch.argmax(batch_strength_probs, dim=1).cpu().numpy()
        tor_strength_predictions.extend(batch_strength_predictions)
        tor_strength_truths.extend(ef_numbers) #+1 because we're converting -1 - 5 to 0 - 6 indexes


        #update visual
        batches_trained += 1
        print("Batch [" + str(batches_trained) + "/" + str(no_batches) + "] Tested")


#model evaluation
print("Calculating evaluation metrics...")

#tornado probability evaluation
accuracy = metrics.accuracy_score(tor_prob_truths, tor_prob_predictions)
precision = metrics.precision_score(tor_prob_truths, tor_prob_predictions)
recall = metrics.recall_score(tor_prob_truths, tor_prob_predictions)
f1 = metrics.f1_score(tor_prob_truths, tor_prob_predictions)
true_negatives, false_positives, false_negatives, true_positives = metrics.confusion_matrix(tor_prob_predictions, tor_prob_truths).ravel()
AUC = metrics.roc_auc_score(tor_prob_truths, tor_prob_predictions)

#tornado intensity evaluation
quad_kappa = metrics.cohen_kappa_score(tor_strength_truths, tor_strength_predictions, weights="quadratic")
summed_truths = Counter(tor_strength_truths)
summed_preds = Counter(tor_strength_predictions)

#nuanced metrics
#nuanced results
total_null = 0
correct_null = 0
total_warned = 0
correct_warned = 0
total_confirmed = 0
correct_confirmed = 0
for category, probability in zip(sample_categories, percentages):
    if (category == 1):
        total_null += 1
        if (probability < 0.3):
            correct_null += 1
    elif (category == 0):
        total_confirmed += 1
        if (probability > 0.6):
            correct_confirmed += 1
    elif (category == 2):
        total_warned += 1
        if (0.2 <= probability):
            correct_warned += 1

#temp histogram plotter
#MODEL_NAME = "TorCast_v2"
#plt.figure(figsize=(8, 5))
#plt.hist(torch.cat(percentages).view(-1).detach().cpu().numpy(), bins=100, range=(0, 1), color="skyblue", edgecolor="black", density=True)
#plt.title("Distribution of " + MODEL_NAME + " predictions")
#plt.xlabel("Prediction")
#plt.ylabel("Density")
#plt.show()


#output
print("\n\n-----------------------------")
print("Results:")
print("-----------------------------")
print("Tornado Probability Head")
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))
print("AUC: " + str(AUC))
print("\nTrue Positives: " + str(true_positives))
print("False Positives: " + str(false_positives))
print("True Negatives: " + str(true_negatives))
print("False Negatives: " + str(false_negatives))
print("-----------------------------")
print("Torando Intensity Head")
print("Quadratic Kappa: " + str(quad_kappa))
print("\nIntensity Counts:")
intensity_labels = ["NonTor", "EF-0", "EF-1", "EF-2", "EF-3", "EF-4", "EF-5"]
reg_tot = 0; reg_pred = 0
sig_tot = 0; sig_pred = 0
for i in range(7):
    print("[" + intensity_labels[i] + "] Actual: " + str(summed_truths[i]) + ", Predicted: " + str(summed_preds[i]))
    if (i == 1 or i == 2 or i == 3):
        reg_tot += summed_truths[i]
        reg_pred += summed_preds[i]
    elif (i == 4 or i == 5 or i == 6):
        sig_tot += summed_truths[i]
        sig_pred += summed_preds[i]
print("Regular Tornado Predictions: " + str(reg_pred) + "/" + str(reg_tot))
print("Significant Tornado Predictions: " + str(sig_pred) + "/" + str(sig_tot))
print("-----------------------------")
print("Nuanced Statistics:")
print("Acceptable Null: " + str(correct_null) + "/" + str(total_null))
print("Acceptable Warned: " + str(correct_warned) + "/" + str(total_warned))
print("Acceptable Confirmed: " + str(correct_confirmed) + "/" + str(total_confirmed))

print("\nTesting completed!")