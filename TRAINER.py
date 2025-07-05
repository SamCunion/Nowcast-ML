#trainer trains the model with training data
import torch
import math
import numpy as np
from load_dataset import get_torcast_dataloader
from TorCastML import TorCastML

#constants
NUM_EPOCHS = 1

data_loader = get_torcast_dataloader("train", 32, 10)

print("Running TorCast Trainer Module")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + "cuda" if torch.cuda.is_available() else "cpu")

model = TorCastML().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_prob = torch.nn.BCELoss()
loss_classifier = torch.nn.CrossEntropyLoss()

print("Beginning TorCastML training for " + str(NUM_EPOCHS) + " epochs...")
for epoch in range(NUM_EPOCHS):
    print("Starting epoch " + str(epoch) + "/" + str(NUM_EPOCHS) + "...")
    model.train()
    percent = 0
    tested = 0
    total = len(data_loader)
    x_per_percent = math.floor(total / 100)
    for batch in data_loader:
        batch_size = len(batch["label"])
        if tested % x_per_percent == 0:
            print("Epoch " + str(percent) + "% complete")
            percent += 1
        tested += 1
        DBZ = batch["DBZ"][...,0].to(DEVICE)
        DBZ = torch.nan_to_num(DBZ, nan=0.0)
        VEL = batch["VEL"][...,0].to(DEVICE)
        VEL = torch.nan_to_num(VEL, nan=0.0)
        RHOHV = batch["RHOHV"][...,0].to(DEVICE)
        RHOHV = torch.nan_to_num(RHOHV, nan=0.0)
        label = batch["label"].to(DEVICE).float()
        ef_number = batch["ef_number"].to(DEVICE).long()

        optimizer.zero_grad()

        prob, class_logits = model(DBZ, VEL, RHOHV)

        #classifier truth
        ef_labels = torch.tensor([batch_size])
        ef_indices = ef_number + 1
        ef_truths = torch.nn.functional.one_hot(ef_indices, num_classes=7).float()
        ef_truths = ef_truths.squeeze(1)
        prob_loss = loss_prob(prob, label)
        class_loss = loss_classifier(class_logits, ef_truths)
        overall_loss = prob_loss + class_loss
        overall_loss.backward()
        optimizer.step()
