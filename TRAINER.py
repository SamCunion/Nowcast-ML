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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + "cuda" if torch.cuda.is_available() else "cpu")

def gaussian_smoothed_loss(logits, targets):
    #compute distance from truth the prediction was
    targets = targets.view(-1).unsqueeze(1).float()
    class_ids = torch.arange(7, device=DEVICE).float().unsqueeze(0)
    dist = (class_ids - targets).abs()

    #gaussian smoothing
    smooth = torch.exp(-0.5 * (dist / CLASS_ACCEPTANCE_METRIC)**2)

    #class imbalance bias
    weighted = smooth * CLASS_WEIGHTS.unsqueeze(0).to(DEVICE)

    #normalise for prob distributions
    soft_targets = weighted / (weighted.sum(dim=1, keepdim=True) + 1e-8)

    #kl divergence loss
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    loss = torch.nn.functional.kl_div(log_probs, soft_targets, reduction="batchmean")
    return loss


#constants
NUM_EPOCHS = 1
OUT_PATH = "./saved_models/"
DATASET_EF_TOTALS = np.array([189275, 5393, 5644, 1997, 651, 172, 1]) #total nontor, ef0, ef1, ef2, ef3, ef4, ef5 (actually 0 ef5, but set to one to avoid divide by zero)
TOTAL_ITEMS = 203132
CLASS_WEIGHTS = torch.tensor(TOTAL_ITEMS / DATASET_EF_TOTALS)
CLASS_ACCEPTANCE_METRIC = 1.0 #how accepting the loss is of "almost right" predictions

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

            if (isinstance(norm_dbz, bool) and norm_dbz == False):
                nan_detected = True
                print("weird dbz matrix detected")
                break;

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
        prob_loss = loss_prob(prob.squeeze(dim=1), label)
        class_loss = gaussian_smoothed_loss(class_logits, ef_indices)
        overall_loss = (prob_loss * 0.5) + (class_loss * 0.5) #scale depending on what head should influence loss more
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