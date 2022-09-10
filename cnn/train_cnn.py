"""
CNN trained on a small imagenet dataset
Imagenette is a subset of 10 easily classified classes from 
Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).
https://github.com/fastai/imagenette
Download the Imagenette dataset from Github to Imageneet folder
"""

from datetime import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import logging as log
import alexnet
import mycnn
import resnet
import os

log.basicConfig(format="%(asctime)s %(message)s", level=log.INFO)


# -------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20  # actual 20 epochs

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    deviceid = torch.cuda.current_device()
    log.info(f"Gpu device {torch.cuda.get_device_name(deviceid)}")


# -------------------------------------------------------------------------------------------------------
# Select the model you want to train
# -------------------------------------------------------------------------------------------------------

modelname = "mycnn_"

if modelname == "mycnn_":
    # Actual image size is 432*320
    model = mycnn.MyCNN().to(device)
    resize_to = transforms.Resize((227, 227))
    
if modelname == "alexnet_":
    # Alexnet model works well for CIFAR-10 when input is scaled to 227x227
    model = alexnet.AlexNet().to(device)
    resize_to = transforms.Resize((227, 227))
    
if modelname == "RestNet50_":
    model = resnet.ResNet50(img_channel=3, num_classes=10).to(device)
    # resizing lower to keep it in memory
    resize_to = transforms.Resize((100, 100))


# -------------------------------------------------------------------------------------------------------
# Load the data from image folder
# -------------------------------------------------------------------------------------------------------

data_dir = "./imagenette2-320"
train_dir = os.path.join(data_dir, "train")
normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


train_transforms = transforms.Compose(
    [resize_to, 
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(), normalize_transform]
)

val_dir = os.path.join(data_dir, "val")

val_transforms = transforms.Compose(
    [resize_to, transforms.ToTensor(), normalize_transform]
)


train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)

val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

# -------------------------------------------------------------------------------------------------------
# Initialise the data loaders
# -------------------------------------------------------------------------------------------------------

workers = 2
pin_memory = True
batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True, #IMPORTANT otherwise the data is not shuffled
    num_workers=workers,
    pin_memory=pin_memory,
    sampler=None,
)

test_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=pin_memory,
)

# initialize our optimizer and loss function
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
lossFn = nn.CrossEntropyLoss()

# -------------------------------------------------------------------------------------------------------
# Train the model
# -------------------------------------------------------------------------------------------------------

for images, labels in train_loader:
    log.info(f"Shape of X [N, C, H, W]: {images.shape}")
    log.info(f"Shape of y: {labels.shape} {labels.dtype}")
    log.info(f"Label: {labels}")
    # test one flow
    # pred = model(x)
    # loss = lossFn(pred, y)
    break
total_step = len(train_loader)
log.info(f"Total steps: {total_step}")

stepsize = total_step // 100
if stepsize < 10:
    stepsize = 10
# loop over our epochs
for epoch in range(0, num_epochs):
    # set the model in training mode
    model.train() #IMPORTANT otherwise the model is not in training mode
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainAccuracy = 0
    totalTrainAccuracy = 0
    valCorrect = 0
    # loop over the training set
    for i, (images, labels) in enumerate(train_loader):
        # send the input to the device
        (images, labels) = (images.to(device), labels.to(device))
        # perform a forward pass and calculate the training loss
        outputs = model(images)
        loss = lossFn(outputs, labels)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad() #IMPORTANT otherwise the gradients of previous batches are not zeroed out
        loss.backward()
        totalTrainLoss += loss
        opt.step()
        # Get the predicted values
        _, predicted = torch.max(outputs.data, 1)
        trainAccuracy = (predicted == labels).float().sum().item()
        trainAccuracy = 100 * trainAccuracy / labels.size(0)
        totalTrainAccuracy += trainAccuracy
        # if (i // stepsize) % 10 == 0:
        log.info(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Accuracy: {:.4f}".format(
                epoch + 1, num_epochs, i + 1, total_step, loss, trainAccuracy
            )
        )

    avgTrainLoss = totalTrainLoss / len(train_loader)
    avgAccuracy = totalTrainAccuracy / len(train_loader)
    log.info(
        "--->Epoch [{}/{}], Average Loss: {:.4f} Average Accuracy: {:.4f}".format(
            epoch + 1, num_epochs, avgTrainLoss, avgAccuracy
        )
    )

# Save the model
torch.save(
    model.state_dict(), modelname + datetime.now().strftime("%H:%M_%B%d%Y") + ".pth"
)

# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    model.eval() #IMPORTANT set model to eval mode before inference
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).float().sum().item()

    print(
        "Accuracy of the network on the {} test images: {} %".format(
            total, 100 * correct / total
        )
    )

    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).float().sum().item()

    print(
        "Accuracy of the network on the {} Train images: {} %".format(
            total, 100 * correct / total
        )
    )

"""
Using MyCNN #32*32

2022-08-09 19:03:59,629 Epoch [20/20], Step [144/148], Loss: 0.0051 Accuracy: 100.0000
2022-08-09 19:03:59,784 Epoch [20/20], Step [145/148], Loss: 0.0013 Accuracy: 100.0000
2022-08-09 19:03:59,940 Epoch [20/20], Step [146/148], Loss: 0.0033 Accuracy: 100.0000
2022-08-09 19:04:00,097 Epoch [20/20], Step [147/148], Loss: 0.0106 Accuracy: 100.0000
2022-08-09 19:04:00,245 Epoch [20/20], Step [148/148], Loss: 0.0006 Accuracy: 100.0000
2022-08-09 19:04:00,313 --->Epoch [20/20], Average Loss: 0.0091 Average Accuracy: 99.7677
Accuracy of the network on the 3925 test images: 36.81528662420382 % --> Bad ?
Accuracy of the network on the 9469 Train images: 99.63037279543775 %

Using AlexNet #227x227

2022-08-08 15:08:48,316 Epoch [20/20], Step [142/148], Loss: 0.8499 Accuracy: 71.8750
2022-08-08 15:08:48,422 Epoch [20/20], Step [143/148], Loss: 0.8604 Accuracy: 68.7500
2022-08-08 15:08:48,507 Epoch [20/20], Step [144/148], Loss: 0.8620 Accuracy: 79.6875
2022-08-08 15:08:48,603 Epoch [20/20], Step [145/148], Loss: 0.6733 Accuracy: 78.1250
2022-08-08 15:08:48,688 Epoch [20/20], Step [146/148], Loss: 0.8089 Accuracy: 75.0000
2022-08-08 15:08:48,786 Epoch [20/20], Step [147/148], Loss: 0.6540 Accuracy: 73.4375
2022-08-08 15:08:48,870 Epoch [20/20], Step [148/148], Loss: 1.0395 Accuracy: 67.2131
2022-08-08 15:08:48,948 --->Epoch [20/20], Average Loss: 0.8318 Average Accuracy: 73.0998
Accuracy of the network on the 3925 test images: 69.78343949044586 % --> Good
Accuracy of the network on the 9469 Train images: 76.4283451262013 %


Using REsnet (50) #100x100 
(reduced image size as was out of memory)

2022-08-08 12:26:48,472 Epoch [20/20], Step [144/148], Loss: 0.1878 Accuracy: 92.1875
2022-08-08 12:26:48,597 Epoch [20/20], Step [145/148], Loss: 0.1052 Accuracy: 96.8750
2022-08-08 12:26:48,723 Epoch [20/20], Step [146/148], Loss: 0.2459 Accuracy: 90.6250
2022-08-08 12:26:48,848 Epoch [20/20], Step [147/148], Loss: 0.1617 Accuracy: 95.3125
2022-08-08 12:26:48,970 Epoch [20/20], Step [148/148], Loss: 0.1481 Accuracy: 95.0820
2022-08-08 12:26:49,055 --->Epoch [20/20], Average Loss: 0.1596 Average Accuracy: 94.3924
Accuracy of the network on the 3925 test images: 69.98726114649682 % ---> Just Good ??
Accuracy of the network on the 9469 Train images: 94.31830182701447 %

"""
