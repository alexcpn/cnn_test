# My CNN Model
# https://alexcpn.medium.com/cnn-from-scratch-b97057d8cef4
# To get the filter use this https://docs.google.com/spreadsheets/d/1tsi4Yl2TwrPg5Ter8P_G30tFLSGQ1i29jqFagxNFa4A/edit?usp=sharing

import torch.nn as nn
import logging as log

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1), # In[3,32,32]
            #nn.BatchNorm2d(6), # Commenting BatchNorm and AvgPool in all the layers does not make any difference
            nn.ReLU(),
            #nn.AvgPool2d(kernel_size = 3, stride = 1),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1), #[6,28,28]
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.AvgPool2d(kernel_size = 3, stride = 1),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=5,stride=1), #[16,24,24] [32,20,20] [C, H,W] 
            # Note when images are added as a batch the size of the output is [N, C, H, W], where N is the batch size ex [1,10,20,20]
            #nn.BatchNorm2d(8),
            nn.ReLU(),
            #nn.AvgPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(in_channels=8,out_channels=4,kernel_size=5,stride=1), # [32,20,20]  [C, H,W] 
            #nn.BatchNorm2d(4),
            nn.ReLU(),
            #nn.AvgPool2d(kernel_size = 3, stride = 2)
            # out #[16,16,16]
        )
        self.linear_stack = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(178084,100),# Note flatten will flatten previous layer output to [N, C*H*W] ex [1,4000]
            nn.ReLU(),
            nn.Linear(100,10)
            
        )
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.cnn_stack(x)
        log.debug("Shape of logits: %s", logits.shape)
        logits = self.flatten(logits)
        log.debug("Shape of logits after flatten: %s", logits.shape) # [N, C*H*W]
        logits = self.linear_stack(logits)
        log.debug("Shape of logits after linear stack: %s", logits.shape) # [N,10]
        #logits = self.softmax(logits) #IMPORTANT: Softmax  is already there in CrossEntropyLoss
        #log.debug("Shape of logits after logSoftmax: %s", logits.shape) #batchsize, 10
        return logits

"""
Without softmax
Epoch [16/20], Step [400/782], Loss: 1.3586
Epoch [17/20], Step [400/782], Loss: 1.4267
Epoch [18/20], Step [400/782], Loss: 1.0434
Epoch [19/20], Step [400/782], Loss: 1.0897
Epoch [20/20], Step [400/782], Loss: 0.7910
Accuracy of the network on the 10000 test images: 55.93 %

Epoch [16/20], Step [400/782], Loss: 0.8177
Epoch [17/20], Step [400/782], Loss: 0.8234
Epoch [18/20], Step [400/782], Loss: 0.8429
Epoch [19/20], Step [400/782], Loss: 0.8053
Epoch [20/20], Step [400/782], Loss: 0.8036
Accuracy of the network on the 10000 test images: 57.93 %

Epoch [14/20], Step [400/782], Loss: 1.1985
Epoch [15/20], Step [400/782], Loss: 1.0795
Epoch [16/20], Step [400/782], Loss: 1.2245
Epoch [17/20], Step [400/782], Loss: 0.8649
Epoch [18/20], Step [400/782], Loss: 1.1103
Epoch [19/20], Step [400/782], Loss: 0.7874
Epoch [20/20], Step [400/782], Loss: 1.1336
Accuracy of the network on the 10000 test images: 59.44 %
"""