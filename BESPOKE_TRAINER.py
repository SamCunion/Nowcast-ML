#customisable trainer that trains until convergence, saving along the way, and can pick up where it left off
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


def main_task(start_epoch, model_name, model):
    #load dataset etc
    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE_NAME)
    CONVERGED = False
    epoch_no = start_epoch
    print("Beginning training " + model_name + " on " + DEVICE_NAME)
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

    while (not CONVERGED):
        epoch_start_time = time.time()
        do_epoch(model, train_data_loader, DEVICE, optimizer, loss_prob, loss_classifier)
        epoch_no += 1
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        acc, prec, rec, f1, qk = do_test(model, test_data_loader, DEVICE)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        qks.append(qks)

        #log metrics
        print("Epoch " + str(epoch_no) + " took " + str(epoch_duration // 3600) + ":" + str((epoch_duration % 3600) // 60) + ":" + str(epoch_duration % 60) + " - " + str(acc) + "," + str(prec) + "," + str(rec) + "," + str(f1) + "," + str(qk))

        #save model
        torch.save(model, MODEL_PATH + "e" + str(epoch_no) + "-" + model_name + ".pt")

        #detect convergence, if the two previous F1 scores are worse than the third previous f1 score, its probably converged
        if ((f1[-3] > f1[-2]) and (f1[-3] > f1[-1])):
            #the model has converged
            CONVERGED = True
        
    #finish up
    print("Model has converged after " + str(epoch_no) + " epochs.")
    input("Press enter to exit...")


def do_epoch(model, data_loader, device, optimizer, loss_prob, loss_classifier):
    model.train()

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
        BATCH_CATEGORY=  batch["category"].squeeze().long()
        SPLIT_STACK = []
        SPLIT_LABEL = []
        SPLIT_EF = []
        SPLIT_CATEGORY = []

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

            #combine scans with the mask to create the stack (DBZ, VEL, RHOHV, MASK)
            STACK = torch.cat([matrices[0], matrices[1], matrices[2], matrices[3]], dim=0)

            SPLIT_STACK.append(STACK)
            SPLIT_LABEL.append(label_data)
            SPLIT_EF.append(ef_data)
            SPLIT_CATEGORY.append(cat_data)
        
        
        #merge batch again
        INPUT_STACK = torch.stack(SPLIT_STACK).to(device)
        LABELS = torch.stack(SPLIT_LABEL).to(device)
        EF_NUMBERS = torch.stack(SPLIT_EF).to(device)
        CATEGORIES = torch.stack(SPLIT_CATEGORY).to(device)

        optimizer.zero_grad()

        prob, class_logits = model(INPUT_STACK)
        
        
        #classifier truth
        ef_indices = EF_NUMBERS + 1
        ef_truths = torch.nn.functional.one_hot(ef_indices, num_classes=7).float()
        ef_truths = ef_truths.squeeze(1)
        prob_loss = loss_prob(prob.squeeze(dim=1), LABELS)
        prob_percent = torch.sigmoid(prob.squeeze(dim=1))
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



def do_test(model, data_loader, device):
    model.eval()

    tor_prob_predictions = []
    tor_prob_truths = []
    tor_strength_predictions = []
    tor_strength_truths = []

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
            SPLIT_STACK = []
            SPLIT_LABEL = []
            SPLIT_EF = []

            for i in range(0, batch_size): #preprocess this item
                dbz_data = BATCH_DBZ[i] #individual dbz input
                vel_data = BATCH_VEL[i] #individual vel input
                rhohv_data = BATCH_RHOHV[i] #individual rhohv input
                label_data = BATCH_LABEL[i] #individual label
                ef_data = BATCH_EF[i] #individual ef number

                matrices = preprocessing_pipeline(dbz_data, vel_data, rhohv_data)
                if (isinstance(matrices, bool) and matrices == False):
                    #invalid, just skip this item in the batch
                    continue

                 #combine scans with the mask to create the stack (DBZ, VEL, RHOHV, MASK)
                STACK = torch.cat([matrices[0], matrices[1], matrices[2], matrices[3]], dim=0)

                SPLIT_STACK.append(STACK)
                SPLIT_LABEL.append(label_data)
                SPLIT_EF.append(ef_data)


            #merge batch again
            INPUT_STACK = torch.stack(SPLIT_STACK).to(device)
            labels = [val.item() for val in SPLIT_LABEL]
            ef_numbers = [int(val.item()) + 1 for val in SPLIT_EF]

            prob, class_logits = model(INPUT_STACK)

            batch_tor_probs = torch.sigmoid(prob)
            batch_tor_predictions = (batch_tor_probs > TORNADO_PROBABILITY_THRESHOLD).int().view(-1).cpu().numpy()
            tor_prob_predictions.extend(batch_tor_predictions)
            tor_prob_truths.extend(labels)

            batch_strength_probs = torch.softmax(class_logits, dim=1)
            batch_strength_predictions = torch.argmax(batch_strength_probs, dim=1).cpu().numpy()
            tor_strength_predictions.extend(batch_strength_predictions)
            tor_strength_truths.extend(ef_numbers) #+1 because we're converting -1 - 5 to 0 - 6 indexes

    #tornado probability evaluation
    accuracy = metrics.accuracy_score(tor_prob_predictions, tor_prob_truths)
    precision = metrics.precision_score(tor_prob_predictions, tor_prob_truths)
    recall = metrics.recall_score(tor_prob_predictions, tor_prob_truths)
    f1 = metrics.f1_score(tor_prob_predictions, tor_prob_truths)

    #tornado intensity evaluation
    quad_kappa = metrics.cohen_kappa_score(tor_strength_truths, tor_strength_predictions, weights="quadratic")

    return accuracy, precision, recall, f1, quad_kappa

    

#entrypoint
if __name__ == "__main__":
    print("TorCastML Bespoke Trainer")
    print("\nNew model, or load model?")

    start_epoch = 0
    model_name = None

    ans = input("New/Load: ")
    if ans.lower() == "new":
        model_name = input("Agent Name: ")
    elif ans.lower() == "load":
        saved_model_list = os.listdir(MODEL_PATH)
        print_str = "Select model to load:\n"
        for i in range(0, len(saved_model_list)):
            print_str += "[" + str(i) + "] " + saved_model_list[i] + "\n"
        print(print_str)
        saved_model_id = int(input("Model ID: "))
        model_name = saved_model_list[saved_model_id]
        MODEL = torch.load(MODEL_PATH + model_name, weights_only=False)
        start_epoch = int(input("Start Epoch: "))
    else:
        exit()