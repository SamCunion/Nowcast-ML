#first deepened and finalised model definition for TorCastML architecture
import torch
import torch.nn as NN

INPUT_CHANNELS = 2 #should always be one, unless stacking input tilts (data + mask)
INPUT_TYPES = 3 #DBZ, VEL, RHOHV
CLASSIFIER_CLASSES = 7 # Nontor, EF0, EF1, EF2, EF3, EF4, EF5

#defines the input segments of the NN
def input_head():
    return NN.Sequential(
        #maybe bigger kernel size better?
        NN.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1, stride=1),
        NN.BatchNorm2d(64),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(64, 128, kernel_size=5, padding=1, stride=1),
        NN.BatchNorm2d(128),
        NN.ReLU(True),
        NN.MaxPool2d(2)
    )

#shared learning, merges the three input heads
def shared_segment():
    return NN.Sequential(
        NN.Conv2d(128 * INPUT_TYPES, 256, kernel_size=7, padding=1, stride=1),
        NN.BatchNorm2d(256),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(256, 256, kernel_size=5, padding=1, stride=1),
        NN.BatchNorm2d(256),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(256, 128, kernel_size=7, padding=1, stride=3),
        NN.BatchNorm2d(128),
        NN.ReLU(True),
        NN.MaxPool2d(2)

    )

#shared fully connected layer(s)
def fc_segment():
    return NN.Sequential(
        NN.Flatten(),

        NN.Linear(2048, 1024),
        NN.Dropout(),
        NN.ReLU(True),

        NN.Linear(1024, 1024),
        NN.Dropout(),
        NN.ReLU(True),

        
    )

#tornado probability head
def prob_head():
    return NN.Sequential(
        NN.Linear(1024, 512),
        NN.ReLU(True),

        NN.Linear(512, 1)
    )

#EF-scale classification head
def intensity_head():
    return NN.Sequential(
        NN.Linear(1024, 1024),
        NN.Dropout(),
        NN.ReLU(True),

        NN.Linear(1024, 512),
        NN.ReLU(True),

        NN.Linear(512, CLASSIFIER_CLASSES)
        #for training, needs raw logits, can apply softmax later
    )

class TorCastML_v1(NN.Module):
    def __init__(self):
        super().__init__()

        #make input heads
        self.DBZ_Head = input_head()
        self.VEL_Head = input_head()
        self.CC_Head = input_head()

        #shared layer(s)
        self.shared = shared_segment()

        #linearise the data
        self.linearise = NN.AdaptiveAvgPool2d((1, 1))

        #FC layer
        self.fc = fc_segment()

        #tornado probability head
        self.tor_prob = prob_head()

        #tornado intensity classifier head
        self.tor_class = intensity_head()
    
    #forward pass through TorCast
    def forward(self, DBZ, VEL, CC):
        #each input is in the form (batch, 1, 240, 120)

        f1 = self.DBZ_Head(DBZ)
        f2 = self.VEL_Head(VEL)
        f3 = self.CC_Head(CC)
        #each is now (batch, 32, 60, 30)

        #merge each head's channels
        combined_heads = torch.cat([f1, f2, f3], dim=1)
        #shape: (batch, 96, 60, 30)
        shared_conv = self.shared(combined_heads)
        #shape: (batch, 64, 30, 15)
        linear = self.linearise(shared_conv)
        #shape: (batch, 64, 1, 1)
        fc = self.fc(linear)
        #shape: (batch, 32)

        #output heads
        out_prob = self.tor_prob(fc) #shape: (batch, 1)
        out_class = self.tor_class(fc) #shape: (batch, CLASSIFIER_CLASSES)

        #simplify dimensions
        out_prob.squeeze(1) #shape: (batch)

        #output
        return out_prob, out_class

