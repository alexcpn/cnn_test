"""
Using Grad CAM libray to visualize the gradients of the last layer
and to see if the model has learned the features of the images
"""

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from importlib.resources import path
import urllib.request
from PIL import Image
from torchvision import transforms
import torch
import resnet
import alexnet
import mycnn
import sys
import numpy as np

import torch.nn as nn

# Test with tench - a fresh water fish
#url_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Tinca_tinca_Prague_Vltava_2.jpg/1920px-Tinca_tinca_Prague_Vltava_2.jpg"
# Test with Church
# path = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Church_of_Saint_Simeon_Stylites_01.jpg/1920px-Church_of_Saint_Simeon_Stylites_01.jpg"
# Test with Truck ( though we have trained on Garbage truck)
# path = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Mitsubishi_Motors_Minicab_MiEV_Truck_%28Prototype%29.jpg/1920px-Mitsubishi_Motors_Minicab_MiEV_Truck_%28Prototype%29.jpg"
# Test with Garbage truck
# path = "https://upload.wikimedia.org/wikipedia/commons/a/aa/US_Garbage_Truck.jpg"


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


# Choose a saved Model - comment out the rest
modelname = "mycnn"

# Choose a saved Model - assign the name you want to test with
# (assuming that you have trained the models)
modelname = "resnet50"
resize_size =(1,1)
if modelname == "mycnn":
    resize_size = (227,227)
    model = mycnn.MyCNN()
    path ="mycnn_13:27_September132022.pth" # Trained with Augmentation
    resize_to = transforms.Resize(resize_size)
if modelname == "alexnet":
    resize_size = (227,227)
    model = alexnet.AlexNet()
    path = "./alexnet_15:08_August082022.pth"
    resize_to = transforms.Resize(resize_size)
if modelname == "resnet50":
    model = resnet.ResNet50(img_channel=3, num_classes=10)
    resize_size =(150,150)
    #path = "./RestNet50_12:26_August082022.pth" # without augumentation
    path = "./RestNet50_13:49_September102022.pth" #with augumentation
    resize_to = transforms.Resize(resize_size)


model.load_state_dict(torch.load(path))
model.eval()

# Set the target for the CAM layer
module_list =[module for module in model.modules()]
print("------------------------")
for count, value in enumerate(module_list):
    print(count, value)
print("------------------------")

print("Last layer of module",module_list[11])
print("Second last layer of module",module_list[8])
     
target_layers = [module_list[8],module_list[11]]

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)


# Load the image
filename = "./test-images/test-truck.jpg"

input_image = Image.open(filename)
preprocess = transforms.Compose(
    [
        resize_to,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # IMPORTANT: normalize for pre-trained models
    ]
)
input_tensor = preprocess(input_image)
print("Input Tensor Shape:", input_tensor.shape)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")
    input_tensor = input_tensor.to("cuda")

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets

# That are, for example, combinations of categories, or specific outputs in a non standard model.
targets = [ClassifierOutputTarget(6)] #0 for finch ?

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_batch, targets=None)
print( "len grayscale_cam",len(grayscale_cam),grayscale_cam.shape)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
from PIL import Image
im = Image.fromarray(grayscale_cam)
if im.mode != 'RGB':
    im = im.convert('L')
im.save("grayscale_cam.jpeg")
img=np.array(input_image.resize(resize_size),np.float32)
print("img shape",img.shape,img.max())
#img = input_image
img *= (1.0/img.max())
#img = Image.fromarray(img)
#img.save("tensortoimage.jpg")
visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
#cam_images = [show_cam_on_image(img, grayscale, use_rgb=True) for img, grayscale in zip(input_image, grayscale_cam)]
visualization = Image.fromarray(visualization)
visualization.save("show_Cam_on_image.jpeg")
sys.exit()



