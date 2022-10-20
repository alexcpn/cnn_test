"""
Utility to check Precision and Recall of a  trained model
Author - Alex Punnen 
"""

from datetime import datetime
from matplotlib.pyplot import bar_label
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
from sklearn.metrics import classification_report, confusion_matrix
import math
import numpy as np
from collections import Counter


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

#precision_per_class = defaultdict(list)
# wrong_per_class = defaultdict(list)
# right_per_class = defaultdict(list)
confusion_matrix = np.zeros((len(categories),len(categories)))

# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    model.eval() #IMPORTANT set model to eval mode before inference
    # correct = 0
    # total = 0


    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # ------------------------------------------------------------------------------------------
        # Predict for the batch of images
        # ------------------------------------------------------------------------------------------
        outputs = model(images)  #Outputs= torch.Size([64, 10]) Probability of each of the 10 classes
        _, predicted = torch.max(outputs.data, 1) # get the class with the highest Probability out Given 1 per image # predicted= torch.Size([64])
        # total += labels.size(0) #labels= torch.Size([64])  This is the truth value per image - the right class
        # correct += (predicted == labels).float().sum().item()  # Find which are correctly classified
        
        # Below illustrates the above Torch Tensor semantics
        # >>> import torch
        # >>> some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
        # >>> some_integers3 = torch.tensor((12, 3, 5, 7, 11, 13, 17, 19))
        # >>> (some_integers ==some_integers3)*(some_integers == 3)
        # tensor([False,  True, False, False, False, False, False, False])
        # >>> ((some_integers ==some_integers3)*(some_integers >12)).sum().item()
        # 3
        
        # ------------------------------------------------------------------------------------------
        #  Lets check also which classes are wrongly predicted with other classes  to create a MultiClass confusion matrix
        # ------------------------------------------------------------------------------------------

        mask=(predicted != labels) # Wrongly predicted
        wrong_predicted =torch.masked_select(predicted,mask)
        wrong_labels =torch.masked_select(labels,mask)
        wrongly_zipped = zip(wrong_labels,wrong_predicted)

        mask=(predicted == labels) # Rightly predicted
        rightly_predicted =torch.masked_select(predicted,mask)
        right_labels =rightly_predicted #same torch.masked_select(labels,mask)
        rightly_zipped = zip(right_labels,rightly_predicted)
        
        # Note that this is for a single batch - add to the list associated with class
        for _,j in enumerate(wrongly_zipped):
            k = j[0].item() # label
            l = j[1].item() # predicted
            #wrong_per_class[k].append(l)
            confusion_matrix[k][l] +=1
       
        # Note that this is for a single batch - add to the list associated with class
        for _,j in enumerate(rightly_zipped):
            k = j[0].item() # label
            l = j[1].item() # predicted
            #right_per_class[k].append(l)
            confusion_matrix[k][l] +=1
    
    #print("Confusion Matrix1=\n",confusion_matrix)
    # ------------------------------------------------------------------------------------------
    # Print Confusion matrix in Pretty print format
    # ------------------------------------------------------------------------------------------
    print(categories)
    for i in range(len(categories)):
        for j in range(len(categories)):
            print(f"\t{confusion_matrix[i][j]}",end='')
        print(f"\t{categories[i]}\n",end='')
    # ------------------------------------------------------------------------------------------
    # Calculate Accuracy per class
    # ------------------------------------------------------------------------------------------
    print("---------------------------------------")
    total_correct =0
    for i in range(len(categories)):
        print(f"Average accuracy per class {categories[i]} from confusion matrix {confusion_matrix[i][i]/confusion_matrix[i].sum()}")
        total_correct +=confusion_matrix[i][i]

    print(f"Average Accuracy?precision from confusion matrix is {total_correct/confusion_matrix.sum()}")

    # Overall accuracy as below
    # print(
    #     "Accuracy of the network on the {} test/validation images: {} %".format(
    #         total, 100 * correct / total
    #     ))
    
    # for key,val in wrong_per_class.items(): # Key is category and val is a list of wrong classes
    #     summed_wrong_classes =Counter(val).most_common()
    #     print(f"**To Predict {categories[key]}")
    #     for ele in summed_wrong_classes:
    #         print(f" --Predicted {categories[ele[0]]} count={ele[1]}")
    #         confusion_matrix[key][ele[0]]=ele[1]

    # for key,val in right_per_class.items(): # Key is category and val is a list of wrong classes
    #     summed_right_classes =Counter(val).most_common()
    #     print(f"**To Predict {categories[key]}")
    #     for ele in summed_right_classes:
    #         print(f" --Predicted {categories[ele[0]]} count={ele[1]}")
    

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
Output 
2022-10-20 13:38:01,112 Gpu device NVIDIA GeForce RTX 3060 Laptop GPU
['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
        320.0   10.0    7.0     25.0    0.0     3.0     3.0     2.0     16.0    1.0     tench
        23.0    286.0   3.0     39.0    3.0     7.0     7.0     3.0     21.0    3.0     English springer
        8.0     4.0     265.0   13.0    2.0     4.0     14.0    36.0    11.0    0.0     cassette player
        49.0    30.0    12.0    194.0   19.0    21.0    27.0    22.0    8.0     4.0     chain saw
        14.0    5.0     10.0    20.0    274.0   21.0    23.0    19.0    16.0    7.0     church
        13.0    21.0    22.0    64.0    26.0    206.0   11.0    19.0    12.0    0.0     French horn
        8.0     4.0     14.0    28.0    23.0    5.0     291.0   14.0    2.0     0.0     garbage truck
        7.0     11.0    50.0    41.0    23.0    9.0     46.0    222.0   8.0     2.0     gas pump
        37.0    38.0    9.0     27.0    17.0    10.0    7.0     6.0     237.0   11.0    golf ball
        10.0    6.0     6.0     46.0    19.0    2.0     13.0    12.0    56.0    220.0   parachute
---------------------------------------
Average accuracy per class tench from confusion matrix 0.8268733850129198
Average accuracy per class English springer from confusion matrix 0.7240506329113924
Average accuracy per class cassette player from confusion matrix 0.742296918767507
Average accuracy per class chain saw from confusion matrix 0.5025906735751295
Average accuracy per class church from confusion matrix 0.6699266503667481
Average accuracy per class French horn from confusion matrix 0.5228426395939086
Average accuracy per class garbage truck from confusion matrix 0.7480719794344473
Average accuracy per class gas pump from confusion matrix 0.5298329355608592
Average accuracy per class golf ball from confusion matrix 0.5939849624060151
Average accuracy per class parachute from confusion matrix 0.5641025641025641
Average Accuracy?precision from confusion matrix is 0.640764331210191
Accuracy of the network on the 3925 test/validation images: 64.07643312101911 %
"""