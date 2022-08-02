"""
 Using the CIFAR-10 dataset on a custom CNN
 Explanation and equations here https://alexcpn.medium.com/cnn-from-scratch-b97057d8cef4
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import logging as log

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)


# My CNN Model
# https://alexcpn.medium.com/cnn-from-scratch-b97057d8cef4
# To get the filter use this https://docs.google.com/spreadsheets/d/1tsi4Yl2TwrPg5Ter8P_G30tFLSGQ1i29jqFagxNFa4A/edit?usp=sharing
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # Input 32*32 * 3
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1), #[6,28,28]
            nn.ReLU(),
            nn.Conv2d(in_channels=6,out_channels=6,kernel_size=15,stride=1), #[6.14.14]
            nn.ReLU(),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1), #[16,10,10] [C, H,W] #changed channels to 10
            # Note when images are added as a batch the size of the output is [N, C, H, W], where N is the batch size ex [1,10,20,20]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=6,stride=1), #[16,5,5] [C, H,W] 
            # Note when images are added as a batch the size of the output is [N, C, H, W], where N is the batch size ex [1,10,20,20]
            nn.ReLU()
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(400,100),# Note flatten will flatten previous layer output to [N, C*H*W] ex [1,4000]
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.logSoftmax = nn.Softmax(dim=1)

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

#-------------------------------------------------------------------------------------------------------
# Code
#-------------------------------------------------------------------------------------------------------

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20 # actual 20 epochs

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    deviceid = torch.cuda.current_device()
    log.info(f"Gpu device {torch.cuda.get_device_name(deviceid)}")

# Load the data

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
# mean calculated like https://stackoverflow.com/a/69750247/429476
all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2470, 0.2435, 0.2616])
                                     #transforms.Normalize(mean=[0.1307], # for MNIST - one channel
                                     #                     std=[0.3081,]) # for MNIST - one channel
                                     ])
# Create Training dataset
train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                             train = True,
                                             transform = all_transforms,
                                             download = True)

# Create Testing dataset
test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                            train = False,
                                            transform = all_transforms,
                                            download=True)

# Instantiate loader objects to facilitate processing
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)



model = MyCNN().to(device)

# initialize our optimizer and loss function
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
lossFn = nn.CrossEntropyLoss()


for x, y in train_loader:
  log.info(f"Shape of X [N, C, H, W]: {x.shape}")
  log.info(f"Shape of y: {y.shape} {y.dtype}")
  # test one flow
  #pred = model(x)
  #loss = lossFn(pred, y)
  break
total_step = len(train_loader)
# loop over our epochs
for epoch in range(0, num_epochs):
  # set the model in training mode
  model.train()
  # initialize the total training and validation loss
  totalTrainLoss = 0
  totalValLoss = 0
  # initialize the number of correct predictions in the training
  # and validation step
  trainCorrect = 0
  valCorrect = 0
	# loop over the training set
  for i, (x, y) in enumerate(train_loader):
    # send the input to the device
    (x, y) = (x.to(device), y.to(device))
    # perform a forward pass and calculate the training loss
    pred = model(x)
    loss = lossFn(pred, y)
    # zero out the gradients, perform the backpropagation step,
    # and update the weights
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (i+1) % 400 == 0:
          print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
  
  # Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


"""
Ouput on MNIST dataset

Epoch [18/20], Step [800/938], Loss: 0.2882
Epoch [19/20], Step [400/938], Loss: 0.1440
Epoch [19/20], Step [800/938], Loss: 0.1439
Epoch [20/20], Step [400/938], Loss: 0.1450
Epoch [20/20], Step [800/938], Loss: 0.1349
Accuracy of the network on the 10000 test images: 88.7 %

Ouput of CIFAT10 dataset

Epoch [17/20], Step [400/782], Loss: 0.0120
Epoch [18/20], Step [400/782], Loss: 0.1071
Epoch [19/20], Step [400/782], Loss: 0.0254
Epoch [20/20], Step [400/782], Loss: 0.0167
Accuracy of the network on the 10000 test images: 50.74 % --> Bad



"""