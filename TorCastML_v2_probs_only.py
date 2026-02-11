#TorCast_v2 architecture, this model stacks the inputs when fed into the model. Processes combined data types for a better spatial understanding of the data.

import torch.nn as NN

INPUT_CHANNELS = 4 #DBZ, VEL, RHOHV, MASK

#defines the input segments of the NN
def conv_layers():
    return NN.Sequential(
        NN.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, padding=1),
        NN.Conv2d(16, 32, kernel_size=3, padding=1),
        NN.BatchNorm2d(32),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(32, 64, kernel_size=3, padding=1),
        NN.BatchNorm2d(64),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
        NN.BatchNorm2d(128),
        NN.ReLU(True),
        NN.MaxPool2d(2),

        NN.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
        NN.BatchNorm2d(256),
        NN.ReLU(True),
        NN.MaxPool2d(2),
    )

#shared fully connected layer(s)
def fc_segment():
    return NN.Sequential(
        NN.Flatten(),

        NN.Linear(256, 128),
        NN.ReLU(True),
        NN.Dropout(0.2),

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

class TorCastML_v2_probs_only(NN.Module):

    #specifies the variables that should be accessed for GRAD-CAM analysis
    gradcam_targets = ["combined_head"]

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

    
    #forward pass through TorCast
    def forward(self, input_stack):
        #shape: (BATCH, 4, 120, 240)

        conv_layer = self.combined_head(input_stack)
        #shape: (BATCH, 256, 30, 60)

        linear = self.linearise(conv_layer)
        #shape: (BATCH, 256, 1, 1)

        fc = self.fc(linear)
        #shape: (BATCH, 64)

        #output heads
        out_prob = self.tor_prob(fc) #shape: (batch, 1)

        #simplify dimensions
        out_prob.squeeze(1) #shape: (batch)

        #output
        return out_prob

