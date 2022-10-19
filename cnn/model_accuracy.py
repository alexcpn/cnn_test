"""
Utility to check Precision and Recall of a  trained model
"""

from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import logging as log
import alexnet
import mycnn
import mycnn2
import resnet
import os
from collections import defaultdict
from sklearn.metrics import classification_report
import math
import numpy as np

log.basicConfig(format="%(asctime)s %(message)s", level=log.INFO)

torch.cuda.empty_cache()

# -------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    deviceid = torch.cuda.current_device()
    log.info(f"Gpu device {torch.cuda.get_device_name(deviceid)}")


# -------------------------------------------------------------------------------------------------------
# Select the model you want to train
# -------------------------------------------------------------------------------------------------------

# Imagenette classes
categories = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]


# Choose a saved Model - assign the name you want to test with
# (assuming that you have trained the models)
modelname = "mycnn2"

if modelname == "mycnn":
    model = mycnn.MyCNN()
    path = "mycnn_18:07_October142022.pth" 
    resize_to = transforms.Resize((227, 227))
if modelname == "mycnn2":
    model = mycnn2.MyCNN2()
    path ="mycnn2_16:43_October182022.pth"
    resize_to = transforms.Resize((227, 227))
if modelname == "alexnet":
    model = alexnet.AlexNet()
    path = "./alexnet_15:08_August082022.pth"
    resize_to = transforms.Resize((227, 227))
if modelname == "resnet50":
    model = resnet.ResNet50(img_channel=3, num_classes=10)
    path = "./RestNet50_11:43_October072022.pth"   # trained with more dog images from imagenet
    resize_to = transforms.Resize((150, 150))


model.load_state_dict(torch.load(path))
model.eval()

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


# -------------------------------------------------------------------------------------------------------
#  Test the model - Find accuracy and per class
# -------------------------------------------------------------------------------------------------------


# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    model.to("cuda")

md = defaultdict(list)

# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    model.eval() #IMPORTANT set model to eval mode before inference
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        #print("Outputs=",outputs.shape) #Outputs= torch.Size([64, 10])
        _, predicted = torch.max(outputs.data, 1) # get the class with the most probability out
        #print("predicted=",predicted.shape,predicted[10]) # predicted= torch.Size([64])
        #print("labels=",labels.shape,labels[10]) #labels= torch.Size([64]) 
        total += labels.size(0)
        correct += (predicted == labels).float().sum().item() # this equals is a tensor type comparison
        #print("correct",correct) # say 56 out of 64
        #print("classification_report",classification_report(labels.cpu(), predicted.cpu()))
        for index, element in enumerate(categories):
            cal = ((predicted == labels)*(labels ==index)).sum().item()/ ((labels == index).sum()) #this is Torch semantics
            # >>> import torch
            # >>> some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
            # >>> some_integers3 = torch.tensor((12, 3, 5, 7, 11, 13, 17, 19))
            # >>> (some_integers ==some_integers3)*(some_integers == 3)
            # tensor([False,  True, False, False, False, False, False, False])
            # >>> ((some_integers ==some_integers3)*(some_integers >12)).sum().item()
            # 3
            if not math.isnan(cal):
                md[element].append(cal.item())
            #print(f"{element}={cal}")

        #break
    avg_accuracy =[]    
    for key,val in md.items():
        avg = np.mean(val)
        md[key] = avg
        avg_accuracy.append(avg)
        print(f"Accuracy of Class {key}={avg}")
    print(f"Average accuracy={np.mean(avg_accuracy)}")

    print(
        "Accuracy of the network on the {} test/validation images: {} %".format(
            total, 100 * correct / total
        )
    )
    

    # correct = 0
    # total = 0
    # for images, labels in train_loader:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     outputs = model(images)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).float().sum().item()
    # # this is not really not needed- but just to cross check if what we calculated during training is accurate
    # print(
    #     "Accuracy of the network on the {} Train images: {} %".format(
    #         total, 100 * correct / total
    #     )
    # )

"""
Output will be similar to

2022-10-19 16:06:28,619 Gpu device NVIDIA GeForce RTX 3060 Laptop GPU
Accuracy of Class tench=0.8504464285714286
Accuracy of Class English springer=0.6907253691128322
Accuracy of Class cassette player=0.7420465648174286
Accuracy of Class chain saw=0.5169889160564968
Accuracy of Class church=0.6264965534210205
Accuracy of Class French horn=0.5337499976158142
Accuracy of Class garbage truck=0.7543565290314811
Accuracy of Class gas pump=0.5343750034059797
Accuracy of Class golf ball=0.5873511944498334
Accuracy of Class parachute=0.5481353274413517
Average accuracy=0.6384671883923666
Accuracy of the network on the 3925 test/validation images: 64.07643312101911 %
"""