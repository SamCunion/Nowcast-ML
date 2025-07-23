#first deepened and finalised model definition for TorCastML architecture
import torch
import torch.nn as NN

INPUT_CHANNELS = 4 #DBZ, VEL, RHOHV, MASK
CLASSIFIER_CLASSES = 7 # Nontor, EF0, EF1, EF2, EF3, EF4, EF5

#defines the input segments of the NN
def conv_layers():
    return NN.Sequential(
        NN.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, padding=1),
        NN.BatchNorm2d(32),
        NN.ReLU(True),

        NN.Conv2d(32, 64, kernel_size=3, padding=1),
        NN.BatchNorm2d(64),
        NN.ReLU(True),

        NN.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
        NN.BatchNorm2d(128),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(128, 256, kernel_size=3, padding=1),
        NN.BatchNorm2d(256),
        NN.ReLU(True),
        NN.MaxPool2d(2)
    )

#shared fully connected layer(s)
def fc_segment():
    return NN.Sequential(
        NN.Flatten(),

        NN.Linear(256, 128),
        NN.ReLU(True),
        NN.Dropout(0.3),

        NN.Linear(128, 64),
        NN.ReLU(True),

    )

#tornado probability head
def prob_head():
    return NN.Sequential(
        NN.Linear(64, 32),
        NN.ReLU(True),

        NN.Linear(32, 1)
    )

#EF-scale classification head
def intensity_head():
    return NN.Sequential(
        NN.Linear(64, 32),
        NN.ReLU(True),

        NN.Linear(32, CLASSIFIER_CLASSES)
        #for training, needs raw logits, can apply softmax later
    )

class TorCastML_v2(NN.Module):
    def __init__(self):
        super().__init__()

        #make conv layers
        self.combined_head = conv_layers()

        #linearise the data
        self.linearise = NN.AdaptiveAvgPool2d((1, 1))

        #FC layer
        self.fc = fc_segment()

        #tornado probability head
        self.tor_prob = prob_head()

        #tornado intensity classifier head
        self.tor_class = intensity_head()
    
    #forward pass through TorCast
    def forward(self, input_stack):
        #input is in the form (batch, 4, 240, 120)

        conv_layer = self.combined_head(input_stack)
        #each is now (batch, 256, 60, 30)

        #shape: (batch, 64, 30, 15)
        linear = self.linearise(conv_layer)
        #shape: (batch, 64, 1, 1)
        fc = self.fc(linear)
        #shape: (batch, 32)

        #output heads
        out_prob = self.tor_prob(fc) #shape: (batch, 1)
        out_class = self.tor_class(fc) #shape: (batch, 7)

        #simplify dimensions
        out_prob.squeeze(1) #shape: (batch)

        #output
        return out_prob, out_class

