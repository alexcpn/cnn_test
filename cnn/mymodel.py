"""
 Coding something similar to LeNet-5 Convolutional Neural Net in PyTorch
 
 
 The Network is as follows:
 
 Input (R,G,B)= [32.32.3] *(5.5.3)*6  == [28.28.6] * (5.5.6)*1 = [24.24.1] *  (5.5.1)*16 = [20.20.16] *
 FC layer 1 (20, 120, 16) == [20,120]* FC layer 2 (120, 1) == [20,1]* FC layer 3 (20, 10) == [10,1]* Softmax  (10,) =(10,1) = Output

 There may be slight changes in the Fully connected layer dimension as batching is involved

 Related repo: https://github.com/alexcpn/cnn_in_python
"""
from torch import nn
import logging as log


log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO) # change to log.DEBUG for more info


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # Input 32*32 * 3
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1), #[6,28,28]
            nn.ReLU(),
            nn.Conv2d(in_channels=6,out_channels=1,kernel_size=5,stride=1), #[1.24.24]
            nn.ReLU(),
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5,stride=1), #[10,20,20] [C, H,W] #changed channels to 10
            # Note when images are added as a batch the size of the output is [N, C, H, W], where N is the batch size ex [1,10,20,20]
            nn.ReLU()
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(4000,10),# Note flatten will flatten previous layer output to [N, C*H*W] ex [1,4000]
            nn.Linear(10,10)
        )
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        logits = self.cnn_stack(x)
        log.debug("Shape of logits: %s", logits.shape)
        logits = self.flatten(logits)
        log.debug("Shape of logits after flatten: %s", logits.shape) # [N, C*H*W]
        logits = self.linear_stack(logits)
        log.debug("Shape of logits after linear stack: %s", logits.shape) # [N,10]
        logits = self.logSoftmax(logits)
        log.debug("Shape of logits after logSoftmax: %s", logits.shape) #batchsize, 10
        return logits