#trainer trains the model with training data
import torch
import math
from load_dataset import get_torcast_dataloader
from TorCastML import TorCastML

#constants
NUM_EPOCHS = 1

data_loader = get_torcast_dataloader("train", 32, 10)
print(data_loader[0])
print("Running TorCast Trainer Module")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + DEVICE)

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
    for DBZ, VEL, RHOHV, label, ef_number in data_loader:
        if tested % x_per_percent == 0:
            print("Epoch " + str(percent) + "\% complete")
            percent += 1
        tested += 1
        DBZ = DBZ[0].to(DEVICE)
        VEL = VEL[0].to(DEVICE)
        RHOHV = RHOHV[0].to(DEVICE)
        label = label.to(DEVICE)
        ef_number = ef_number.to(DEVICE)

        optimizer.zero_grad()

        prob, class_logits = model(DBZ, VEL, RHOHV)

        prob_loss = loss_prob(prob, label)
        class_loss = loss_classifier(class_logits, ef_number)
        overall_loss = prob_loss + class_loss
        overall_loss.backward()
        optimizer.step()
