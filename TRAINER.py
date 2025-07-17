#trainer trains the model with training data
import torch
import time
import random
import numpy as np
from load_dataset import get_torcast_dataloader
from TorCastML import TorCastML
from input_preprocessing import preprocessing_pipeline

#entry
print("TorCast trainer module")
print("begin training?")
inp = input("Y/N: ")
if inp.lower() != "y":
    exit()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + "cuda" if torch.cuda.is_available() else "cpu")

#constants
NUM_EPOCHS = 1
OUT_PATH = "./saved_models/"
DATASET_EF_TOTALS = np.array([189275, 5393, 5644, 1997, 651, 172, 1]) #total nontor, ef0, ef1, ef2, ef3, ef4, ef5 (actually 0 ef5, but set to one to avoid divide by zero)
TOTAL_ITEMS = 203132

data_loader = get_torcast_dataloader("train", 32, 10)

print("Running TorCast Trainer Module")

model = TorCastML().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_prob = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([94 / 6]).to(DEVICE)) #biases loss towards positives, 6% are positives according to TorNet
loss_classifier = torch.nn.CrossEntropyLoss(weight=torch.tensor((TOTAL_ITEMS / DATASET_EF_TOTALS), dtype=torch.float32).to(DEVICE)) #biases classifier since very few tornado examples exist

print("Beginning TorCastML training for " + str(NUM_EPOCHS) + " epochs...")
process_start_time = time.time()
for epoch in range(NUM_EPOCHS):
    print("Starting epoch " + str(epoch) + "/" + str(NUM_EPOCHS) + "...")
    model.train()
    no_batches = len(data_loader)
    batches_trained = 0
    epoch_start_time = time.time()
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

            matrices = preprocessing_pipeline(dbz_data, vel_data, rhohv_data)
            if (isinstance(matrices, bool) and matrices == False):
                #invalid, just skip this item in the batch
                continue

            SPLIT_DBZ.append(matrices[0])
            SPLIT_VEL.append(matrices[1])
            SPLIT_RHOHV.append(matrices[2])
        
        
        #merge batch again
        DBZ = torch.stack(SPLIT_DBZ).to(DEVICE)
        VEL = torch.stack(SPLIT_VEL).to(DEVICE)
        RHOHV = torch.stack(SPLIT_RHOHV).to(DEVICE)



        label = batch["label"].squeeze().to(DEVICE).float()
        ef_number = batch["ef_number"].squeeze().to(DEVICE).long()


        optimizer.zero_grad()

        prob, class_logits = model(DBZ, VEL, RHOHV)
        
        
        #classifier truth
        ef_labels = torch.tensor([batch_size])
        ef_indices = ef_number + 1
        ef_truths = torch.nn.functional.one_hot(ef_indices, num_classes=7).float()
        ef_truths = ef_truths.squeeze(1)
        prob_loss = loss_prob(prob.squeeze(dim=1), label)
        class_loss = loss_classifier(class_logits, ef_truths)
        overall_loss = (prob_loss * 0.7) + (class_loss * 0.3) #scale depending on what head should influence loss more
        overall_loss.backward()
        optimizer.step()

        #update visual
        batches_trained += 1
        print("Epoch [" + str(epoch + 1) + "/" + str(NUM_EPOCHS) + "] Batch [" + str(batches_trained) + "/" + str(no_batches) + "] Trained")
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print("=====================================")
    print("Epoch " + str(epoch + 1) + " took " + str(epoch_duration // 3600) + "h, " + str((epoch_duration % 3600) // 60) + "m, " + str(epoch_duration % 60) + "s")
    if epoch == 0:
        print("Training should take approximately " + str((epoch_duration * (NUM_EPOCHS - 1)) // 3600) + "h, " + str(((epoch_duration * (NUM_EPOCHS - 1)) % 3600) // 60) + "m, " + str((epoch_duration * (NUM_EPOCHS - 1)) % 60) + "s")
    print("=====================================")

process_end_time = time.time()
process_duration = process_end_time - process_start_time
print("=====================================")
print("=====================================")
print("TRAINING PROCESS COMPLETE")
print("Model training took " + str(process_duration // 3600) + "h, " + str((process_duration % 3600) // 60) + "m, " + str(process_duration % 60) + "s")

print("Save the model?")
inp = input("Y/N: ")
if inp.lower() != "y":
    exit()

inp = input("Model name: ")
print("saving model...")
torch.save(model, OUT_PATH + "TorCastML-" + inp + ".pt")
print("model saved!")