#trainer trains the model with training data
import torch
import time
import numpy as np
from load_dataset import get_torcast_dataloader
from TorCastML import TorCastML
from input_preprocessing import normalise_input

#entry
print("TorCast trainer module")
print("begin training?")
inp = input("Y/N: ")
if inp.lower() != "y":
    exit()

#constants
NUM_EPOCHS = 3
OUT_PATH = "./saved_models/"

data_loader = get_torcast_dataloader("train", 32, 10)

print("Running TorCast Trainer Module")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + "cuda" if torch.cuda.is_available() else "cpu")

model = TorCastML().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_prob = torch.nn.BCEWithLogitsLoss()
loss_classifier = torch.nn.CrossEntropyLoss()

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

        nan_detected = False
        for i in range(0, batch_size): #preprocess this item
            dbz_data = BATCH_DBZ[i] #individual dbz input
            vel_data = BATCH_VEL[i] #individual vel input
            rhohv_data = BATCH_RHOHV[i] #individual rhohv input

            #is one of the inputs filled with nans? red alert!
            if (torch.isnan(dbz_data).all() or torch.isnan(vel_data).all() or torch.isnan(rhohv_data).all()):
                nan_detected = True
                break;

            norm_dbz = normalise_input("DBZ", dbz_data)
            norm_vel = normalise_input("VEL", vel_data)
            norm_rhohv = normalise_input("RHOHV", rhohv_data)

            SPLIT_DBZ.append(norm_dbz)
            SPLIT_VEL.append(norm_vel)
            SPLIT_RHOHV.append(norm_rhohv)
        
        if (nan_detected): #discard the batch to prevent dirty data
            print("BATCH CONTAINED ALL NAN DATA!")
            continue
        
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
        overall_loss = prob_loss + class_loss
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