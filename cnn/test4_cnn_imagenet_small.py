"""
CNN trained on a small imagenet dataset
Imagenette is a subset of 10 easily classified classes from 
Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).
https://github.com/fastai/imagenette
Download the Imagenette dataset from Github to Imageneet folder
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import logging as log
import alexnet
import mycnn
import os

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)


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


#-------------------------------------------------------------------------------------------------------
# Load the model
#-------------------------------------------------------------------------------------------------------

# Alexnet model works well for CIFAR-10 when input is scaled to 227x227 (from 32x32)
model = alexnet.AlexNet().to(device)
#model = mycnn.MyCNN().to(device)

#-------------------------------------------------------------------------------------------------------
# Load the data from image folder
#-------------------------------------------------------------------------------------------------------

data_dir = './imagenette2-320'
train_dir = os.path.join(data_dir, 'train')
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((426, 320)),
    transforms.ToTensor(),
    normalize_transform
])

val_dir = os.path.join(data_dir, 'val')

val_transforms = transforms.Compose([
        transforms.Resize((426, 320)),
        transforms.ToTensor(),
        normalize_transform
    ])
    


train_dataset = torchvision.datasets.ImageFolder(
    train_dir,
    train_transforms
)

val_dataset = torchvision.datasets.ImageFolder(
        val_dir,
        val_transforms
    )

#-------------------------------------------------------------------------------------------------------
# Initialise the dataloaders
#-------------------------------------------------------------------------------------------------------

workers = 2
pin_memory = True
batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=pin_memory,
    sampler=None
)

test_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=pin_memory
)

# initialize our optimizer and loss function
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
lossFn = nn.CrossEntropyLoss()

#-------------------------------------------------------------------------------------------------------
# Train the model
#-------------------------------------------------------------------------------------------------------

for images, labels in train_loader:
  log.info(f"Shape of X [N, C, H, W]: {images.shape}")
  log.info(f"Shape of y: {labels.shape} {labels.dtype}")
  log.info(f"Label: {labels}")
  # test one flow
  #pred = model(x)
  #loss = lossFn(pred, y)
  break
total_step = len(train_loader)
log.info(f"Total steps: {total_step}")

stepsize = total_step // 100
if stepsize < 10:
    stepsize = 10
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
    if (i+1) % stepsize == 0:
          log.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
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
First output: 227x227

Epoch [20/20], Step [144/148], Loss: 1.0208
Epoch [20/20], Step [145/148], Loss: 0.7248
Epoch [20/20], Step [146/148], Loss: 0.8423
Epoch [20/20], Step [147/148], Loss: 0.9355
Epoch [20/20], Step [148/148], Loss: 0.9529
Accuracy of the network on the 10000 test images: 29.070063694267517 %

Second output with original image size: 320x426
2022-08-04 19:26:18,503 Epoch [20/20], Step [40/148], Loss: 1.5144
2022-08-04 19:26:20,681 Epoch [20/20], Step [50/148], Loss: 1.7784
2022-08-04 19:26:22,865 Epoch [20/20], Step [60/148], Loss: 1.6875
2022-08-04 19:26:25,064 Epoch [20/20], Step [70/148], Loss: 1.8566
2022-08-04 19:26:27,261 Epoch [20/20], Step [80/148], Loss: 1.7832
2022-08-04 19:26:29,443 Epoch [20/20], Step [90/148], Loss: 1.7009
2022-08-04 19:26:31,623 Epoch [20/20], Step [100/148], Loss: 1.5809
2022-08-04 19:26:33,821 Epoch [20/20], Step [110/148], Loss: 1.9113
2022-08-04 19:26:36,018 Epoch [20/20], Step [120/148], Loss: 1.8923
2022-08-04 19:26:38,200 Epoch [20/20], Step [130/148], Loss: 1.7649
2022-08-04 19:26:40,399 Epoch [20/20], Step [140/148], Loss: 1.7015
Accuracy of the network on the 10000 test images: 16.89171974522293 %
"""