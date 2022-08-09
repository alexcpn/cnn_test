"""
Pre trained  model from tutorial modified from
https://pytorch.org/hub/pytorch_vision_alexnet/
And for imagenette small dataset
Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).
https://github.com/fastai/imagenette

Pre-trained model from test4_cnn_imagenet_small.py in the same folder
"""

from importlib.resources import path
import urllib.request
from PIL import Image
from torchvision import transforms
import torch
import resnet
import alexnet
import mycnn

# Test with tench - a fresh water fish
# path = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Tinca_tinca_Prague_Vltava_2.jpg/1920px-Tinca_tinca_Prague_Vltava_2.jpg"
# Test with Church
path = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Church_of_Saint_Simeon_Stylites_01.jpg/1920px-Church_of_Saint_Simeon_Stylites_01.jpg"
# Test with Truck ( though we have trained on Garbage truck)
path = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Mitsubishi_Motors_Minicab_MiEV_Truck_%28Prototype%29.jpg/1920px-Mitsubishi_Motors_Minicab_MiEV_Truck_%28Prototype%29.jpg"
# Test with Garbage truck
path = "https://upload.wikimedia.org/wikipedia/commons/a/aa/US_Garbage_Truck.jpg"
url, filename = path, "test.jpg"

try:
    urllib.request.urlopen(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

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

if modelname == "mycnn":
    model = mycnn.MyCNN()
    path = "./mycnn_15:14_August082022.pth"
    resize_to = transforms.Resize((32, 32))
if modelname == "alexnet":
    model = alexnet.AlexNet()
    path = "./alexnet_15:08_August082022.pth"
    resize_to = transforms.Resize((227, 227))
if modelname == "resnet50":
    model = resnet.ResNet50(img_channel=3, num_classes=10)
    path = "./RestNet50_12:26_August082022.pth"
    resize_to = transforms.Resize((100, 100))


model.load_state_dict(torch.load(path))
model.eval()

input_image = Image.open(filename)
preprocess = transforms.Compose(
    [
        resize_to,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # IMPORTANT: normalize for pretrained models
    ]
)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    output = model(input_batch)
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

"""
Output Resnet50 -Works Great

tench 0.9974397420883179
English springer 0.0012430829228833318
golf ball 0.0011808514827862382
chain saw 4.901618740404956e-05
French horn 3.386243406566791e-05

church 0.9996765851974487
cassette player 0.00014972625649534166
tench 7.752762030577287e-05
French horn 2.9099957828293554e-05
golf ball 2.4453636797261424e-05

Test on a normal truck - Somehow Resnet is not able to generalize well ??

Tested with resize on 100 * 100 (on which the network was trained)

cassette player 0.9416605830192566 --> not good
garbage truck 0.05656484141945839
French horn 0.000811699777841568
English springer 0.0005154424579814076
gas pump 0.00035581536940298975

Test on Actual Garbage truck

garbage truck 0.9998109936714172
cassette player 8.67886483320035e-05
gas pump 8.608541247667745e-05
church 1.5945111954351887e-05
golf ball 1.385972154821502e-07

---------------------------
Output Alexnet - Works okay
----------------------------

tench 0.7363258600234985
golf ball 0.1412317305803299
cassette player 0.044817518442869186
chain saw 0.02296517603099346
English springer 0.01489443145692348

church 0.947597086429596
chain saw 0.012251818552613258
French horn 0.011542724445462227
garbage truck 0.009849738329648972
parachute 0.008836846798658371

Test on a normal truck (not in training)
garbage truck 0.7310217022895813  --> Good
gas pump 0.09301765263080597
chain saw 0.05727909877896309
cassette player 0.05487482622265816
church 0.02958359383046627

Test on Actual Garbage truck

gas pump 0.3171851336956024  --> Not good
garbage truck 0.25640568137168884 --> but close
chain saw 0.13151127099990845
church 0.10047641396522522
cassette player 0.08960364013910294

Ouput MyCNN - Not Bad!

tench 0.9997403025627136
garbage truck 0.00014096715312916785
chain saw 4.286440525902435e-05
parachute 2.5661303880042396e-05
cassette player 1.4581252798961941e-05

church 0.6350839734077454
gas pump 0.23613542318344116
cassette player 0.03102090395987034
garbage truck 0.026711691170930862
French horn 0.025713106617331505

Test on a normal truck (not in training)

garbage truck 0.3535389304161072 --> Score is low but not bad
English springer 0.2765689790248871
gas pump 0.177799791097641
cassette player 0.0679362341761589
church 0.06341461837291718

Test on Actual Garbage truck

garbage truck 0.5828292965888977 --> not bad
cassette player 0.11013609915971756
gas pump 0.09917507320642471
golf ball 0.0536833293735981
French horn 0.04130929708480835

"""
