#Continuous model trainer, uses the complete training set, and then the complete testing set to evaluate the performance of the model after each epoch
#outputs ACC/PREC/REC/F1/QK, as well as TP/FP/TN/FN

import torch
import time
import numpy as np
import os;
from load_dataset import get_torcast_dataloader
from TorCastML_v0 import TorCastML_v0
from TorCastML_v1 import TorCastML_v1
from TorCastML_v2 import TorCastML_v2
from input_preprocessing import preprocessing_pipeline
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", message="The given NumPy array is not writable*")

#==============================================================================================================
#Overall Settings
MODEL = TorCastML_v2() #define the model to train
MODEL_PATH = "./saved_models/bespoke/" #saved models directory
#==============================================================================================================
#Training Settings
PROBABILITY_WARNING_FORGIVENESS = 0.2 #scales loss by this amount when predicting true on warned, but unconfirmed tornado
DATASET_EF_TOTALS = np.array([189275, 5393, 5644, 1997, 651, 172, 1]) #total nontor, ef0, ef1, ef2, ef3, ef4, ef5 (actually 0 ef5, but set to one to avoid divide by zero)
TOTAL_ITEMS = 203132
#==============================================================================================================
#Testing Settings
TORNADO_PROBABILITY_THRESHOLD = 0.5 #tornado prediction probability threshold, above this value is considered an identification of a tornado
#==============================================================================================================

#takes the starting epoch (the number of the last epoch on a loaded model, otherwise 0)
#model_name is the string name of the model
#model is a TorCast model with params already loaded, or randomly initialised
def main_task(start_epoch, model_name, model):
    #load dataset etc
    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE_NAME)
    CONVERGED = False
    epoch_no = start_epoch
    print("Beginning training of '" + model_name + "' on " + DEVICE_NAME)
    
    #get the training and testing sets with batch size 32 and 10 workers
    train_data_loader = get_torcast_dataloader("train", 32, 10)
    test_data_loader = get_torcast_dataloader("test", 32, 10)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_prob = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([94 / 6]).to(DEVICE), reduction="none") #biases loss towards positives, 6% are positives according to TorNet
    loss_classifier = torch.nn.CrossEntropyLoss(weight=torch.tensor((TOTAL_ITEMS / DATASET_EF_TOTALS), dtype=torch.float32).to(DEVICE)) #biases classifier since very few tornado examples exist

    accs = []
    precs = []
    recs = []
    f1s = []
    qks = []

    #training/testing loop
    while (not CONVERGED):
        #TRAIN
        epoch_start_time = time.time()
        do_epoch(model, train_data_loader, DEVICE, optimizer, loss_prob, loss_classifier)
        epoch_no += 1
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        #TEST
        acc, prec, rec, f1, qk, tp, fp, tn, fn = do_test(model, test_data_loader, DEVICE)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        qks.append(qks)
        

        #log metrics
        print("e" + str(epoch_no) + " took " + str((round(epoch_duration % 3600)) // 60) + ":" + str(round(epoch_duration % 60)) + " - " + str(round(acc, 4)) + "," + str(round(prec, 4)) + "," + str(round(rec, 4)) + "," + str(round(f1, 4)) + "," + str(round(qk, 4)) + " | " + str(tp) + "," + str(fp) + "," + str(tn) + "," + str(fn))

        #save model
        torch.save(model, MODEL_PATH + "e" + str(epoch_no) + "-" + model_name + ".pt")


#trains the model on the entire test dataset
#pytorch model, tornet dataloader, pytorch device, optimizer, loss function of the tornado probability head, loss function of the intensity classifier head
def do_epoch(model, data_loader, device, optimizer, loss_prob, loss_classifier):
    model.train()

    for batch in data_loader:
        batch_size = len(batch["label"])
        
        #PREPROCESSING

        #separate batch into items
        BATCH_DBZ = batch["DBZ"][...,0]
        BATCH_VEL = batch["VEL"][...,0]
        BATCH_RHOHV = batch["RHOHV"][...,0]
        BATCH_LABEL = batch["label"].squeeze().float()
        BATCH_EF = batch["ef_number"].squeeze().long()
        BATCH_CATEGORY=  batch["category"].squeeze().long()
        SPLIT_STACK = []
        SPLIT_DBZ = []
        SPLIT_VEL = []
        SPLIT_RHOHV = []
        SPLIT_LABEL = []
        SPLIT_EF = []
        SPLIT_CATEGORY = []

        for i in range(0, batch_size): #preprocess individual sample within batch
            dbz_data = BATCH_DBZ[i] #sample dbz input
            vel_data = BATCH_VEL[i] #sample vel input
            rhohv_data = BATCH_RHOHV[i] #sample rhohv input
            label_data = BATCH_LABEL[i] #sample label
            ef_data = BATCH_EF[i] #sample ef number
            cat_data = BATCH_CATEGORY[i] #sample category label

            #perform preprocessing on dbz, vel, rhohv
            matrices = preprocessing_pipeline(dbz_data, vel_data, rhohv_data)

            if (isinstance(matrices, bool) and matrices == False):
                #invalid, just skip this item in the batch
                continue
            
            #FOR TorCast v1
            #combine proessed matrices with mask
            MASKED_DBZ = torch.cat([matrices[0], matrices[3]], dim=0)
            MASKED_VEL = torch.cat([matrices[1], matrices[3]], dim=0)
            MASKED_RHOHV = torch.cat([matrices[2], matrices[3]], dim=0)
            SPLIT_DBZ.append(MASKED_DBZ)
            SPLIT_VEL.append(MASKED_VEL)
            SPLIT_RHOHV.append(MASKED_RHOHV)

            ##FOR TorCast v2:
            #combine scans with the mask to create the stack (DBZ, VEL, RHOHV, MASK)
            #STACK = torch.cat([matrices[0], matrices[1], matrices[2], matrices[3]], dim=0)
            #SPLIT_STACK.append(STACK)

            SPLIT_LABEL.append(label_data)
            SPLIT_EF.append(ef_data)
            SPLIT_CATEGORY.append(cat_data)
        
        
        #merge batch again
        #FOR TorCast v1:
        DBZ = torch.stack(SPLIT_DBZ, dim=0).to(device)
        VEL = torch.stack(SPLIT_VEL, dim=0).to(device)
        RHOHV = torch.stack(SPLIT_RHOHV, dim=0).to(device)

        ##FOR TorCast v2:
        #INPUT_STACK = torch.stack(SPLIT_STACK).to(device)

        LABELS = torch.stack(SPLIT_LABEL).to(device)
        EF_NUMBERS = torch.stack(SPLIT_EF).to(device)
        CATEGORIES = torch.stack(SPLIT_CATEGORY).to(device)

        #reset gradients in optimizer
        optimizer.zero_grad()

        #change inputs depending on v1 or v2
        prob_logit, class_logits = model(DBZ, VEL, RHOHV)
        
        
        #convert EF rating to classifier indexes
        ef_indices = EF_NUMBERS + 1
        #convert EF rating to truthful index labels
        ef_truths = torch.nn.functional.one_hot(ef_indices, num_classes=7).float()
        ef_truths = ef_truths.squeeze(1)

        #get the loss from the tornado probability head, and the "percentage" chance of tornado
        prob_loss = loss_prob(prob_logit.squeeze(dim=1), LABELS)
        prob_percent = torch.sigmoid(prob_logit.squeeze(dim=1))

        #get the loss from the intensity classifier head
        class_loss = loss_classifier(class_logits, ef_truths)

        #if probability head decided "yes" and type is "warned", scale head loss by defined amount
        #if prediction is "yes" (> 0.5) and label is 0 (no confirmed torndado) and category is 2 (warned)
        leniency_matches = (prob_percent > 0.5) & (LABELS == 0) & (CATEGORIES == 2)
        leniency_scale = torch.ones_like(LABELS, dtype=torch.float32)
        leniency_scale[leniency_matches] = PROBABILITY_WARNING_FORGIVENESS
        prob_loss *= leniency_scale
        prob_loss = prob_loss.mean()

        overall_loss = (prob_loss * 0.7) + (class_loss * 0.3) #scale depending on which head should influence loss more
        overall_loss.backward()
        optimizer.step()


#tests on the entire test set, and returns ACC/PREC/REC/F1 and confusion matrix counts
#inputs pytorch model, test set dataloader, pytorch device
def do_test(model, data_loader, device):
    model.eval()

    tor_prob_predictions = []
    tor_prob_truths = []
    tor_strength_predictions = []
    tor_strength_truths = []

    with torch.no_grad():
        for batch in data_loader:
            batch_size = len(batch["label"])

            #PREPROCESSING

            #separate batch into items
            BATCH_DBZ = batch["DBZ"][...,0]
            BATCH_VEL = batch["VEL"][...,0]
            BATCH_RHOHV = batch["RHOHV"][...,0]
            BATCH_LABEL = batch["label"].squeeze().float()
            BATCH_EF = batch["ef_number"].squeeze().long()
            SPLIT_STACK = []
            SPLIT_DBZ = []
            SPLIT_VEL = []
            SPLIT_RHOHV = []
            SPLIT_LABEL = []
            SPLIT_EF = []

            for i in range(0, batch_size): #preprocess this sample
                dbz_data = BATCH_DBZ[i] #sample dbz input
                vel_data = BATCH_VEL[i] #sample vel input
                rhohv_data = BATCH_RHOHV[i] #sample rhohv input
                label_data = BATCH_LABEL[i] #sample label
                ef_data = BATCH_EF[i] #sample ef number

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


            #merge batch again
            #FOR v1:
            DBZ = torch.stack(SPLIT_DBZ, dim=0).to(device)
            VEL = torch.stack(SPLIT_VEL, dim=0).to(device)
            RHOHV = torch.stack(SPLIT_RHOHV, dim=0).to(device)

            ##FOR v2:
            #INPUT_STACK = torch.stack(SPLIT_STACK).to(device)

            labels = [val.item() for val in SPLIT_LABEL]
            ef_numbers = [int(val.item()) + 1 for val in SPLIT_EF]

            #change inputs depending on v1 or v2
            prob_logit, class_logits = model(DBZ, VEL, RHOHV)

            #collects the batch's tornado probabilities, and thresholds them to separate into positive and negative identifications
            batch_tor_probs = torch.sigmoid(prob_logit)
            batch_tor_predictions = (batch_tor_probs > TORNADO_PROBABILITY_THRESHOLD).int().view(-1).cpu().numpy()
            tor_prob_predictions.extend(batch_tor_predictions)
            tor_prob_truths.extend(labels)

            #collects the batch's tornado intensity probabilities, and gets each sample's strongest predicted class
            batch_strength_probs = torch.softmax(class_logits, dim=1)
            batch_strength_predictions = torch.argmax(batch_strength_probs, dim=1).cpu().numpy()
            tor_strength_predictions.extend(batch_strength_predictions)
            tor_strength_truths.extend(ef_numbers) #+1 because we're converting -1 - 5 to 0 - 6 indexes

    #tornado probability evaluation
    accuracy = metrics.accuracy_score(tor_prob_truths, tor_prob_predictions)
    precision = metrics.precision_score(tor_prob_truths, tor_prob_predictions)
    recall = metrics.recall_score(tor_prob_truths, tor_prob_predictions)
    f1 = metrics.f1_score(tor_prob_truths, tor_prob_predictions)
    true_negatives, false_positives, false_negatives, true_positives = metrics.confusion_matrix(tor_prob_predictions, tor_prob_truths).ravel()

    #tornado intensity evaluation
    quad_kappa = metrics.cohen_kappa_score(tor_strength_truths, tor_strength_predictions, weights="quadratic")

    return accuracy, precision, recall, f1, quad_kappa, true_positives, false_positives, true_negatives, false_negatives

    

#entrypoint
if __name__ == "__main__":
    print("TorCastML Bespoke Trainer")
    print("\nNew model, or load model?")

    start_epoch = 0
    model_name = None

    ans = input("New/Load: ")
    if ans.lower() == "new": #uses the default model loaded earlier, with randomised weights. Starts at epoch 0 (1)
        model_name = input("Agent Name: ")
        main_task(start_epoch, model_name, MODEL)
    elif ans.lower() == "load": #reads the directory, creates a simple file selector interface
        saved_model_list = os.listdir(MODEL_PATH)
        print_str = "Select model to load:\n"
        for i in range(0, len(saved_model_list)):
            print_str += "[" + str(i) + "] " + saved_model_list[i] + "\n"
        print(print_str)
        saved_model_id = int(input("Model ID: "))
        model_name = saved_model_list[saved_model_id]
        MODEL = torch.load(MODEL_PATH + model_name, weights_only=False) #load the entire saved model params
        start_epoch = int(input("Last Epoch: "))
        main_task(start_epoch, model_name, MODEL) #starts with the loaded model, and defined starting epoch
    else:
        exit()