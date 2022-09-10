# Test's with Convolutional Neural Network

This contains a simple custom CNN model `mycnn*.py` and other simpler and older models like AlexNet and ResNet50 using PyTorch

This repo is to understand each model. Also understanding CNN in depth via training and testing these models with different images 

## Training

Training is happening in [cnn/train_cnn.py](cnn/train_cnn.py) script

To train a small subset of Imagenet is used called Imagenette; It is a subset of 10 easily classified classes from 
Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).

The data is linked here https://github.com/fastai/imagenette

Models trained are saved and uploaded here https://drive.google.com/drive/folders/1e70lEahZkiEf8Y-4BHYa6LGMYj23t10p?usp=sharing. 

Note that the saved models are put in the commit hash of the model file. This is to keep track of models and their trainings. Example 
https://drive.google.com/drive/u/0/folders/1wsBnfy09egmORLMe6jDnchIwF0ShkBH6 --> mycnn.py's commit hash `1wsBnfy..`

I am running the training and inference in GPU in a RTX3060 Laptop (Acer Nitro5) with PopOS (from System32) and their way of installing CUDA and NVIDIA drivers. (Doing it manually with CUDA run files in Ubuntu/other distro's is arduous as it affects the display if something is wrong)

Accuracy of MyCNN against Validation is pretty bad though with Train it is good; Meaning it is overfitting badly. But surprisingly  it is able to generalize a Truck image in testing as a Garbage truck, though ResNet50 is not.

Which means that if we change the way of training; a set of smaller CNNS may be as accurate as bigger ones?

```
MyCNN

2022-08-09 19:04:00,245 Epoch [20/20], Step [148/148], Loss: 0.0006 Accuracy: 100.0000
2022-08-09 19:04:00,313 --->Epoch [20/20], Average Loss: 0.0091 Average Accuracy: 99.7677
Accuracy of the network on the 3925 test images: 36.81528662420382 % --> Bad ?
Accuracy of the network on the 9469 Train images: 99.63037279543775 %

Resnet50

2022-08-08 12:26:48,970 Epoch [20/20], Step [148/148], Loss: 0.1481 Accuracy: 95.0820
2022-08-08 12:26:49,055 --->Epoch [20/20], Average Loss: 0.1596 Average Accuracy: 94.3924
Accuracy of the network on the 3925 test images: 69.98726114649682 % ---> Just Good ??
Accuracy of the network on the 9469 Train images: 94.31830182701447 %
```

## Testing

Testing is happening in [cnn/test_cnn.py](cnn/test_cnn.py)

The trained models are tested with some test images pulled from the internet.

Example is test with a similar but different class. In Imagenette there is only Garbage Truck class. However when I test with a normal truck, the best network here (comparatively the most modern -Resnet50) thinks it is a cassette player ! though my few layers imple CNN with no Dropout or MaxPooling get's that right.


```
-------------------------------
Detecting for class test-garbagetruck.jpg model resnet50
--------------------------------
garbage truck 0.9998109936714172
cassette player 8.67886483320035e-05
gas pump 8.608541247667745e-05
church 1.5945111954351887e-05
golf ball 1.385972154821502e-07
--------------------------------
Detecting for class test-truck.jpg model resnet50
--------------------------------
cassette player 0.9416605830192566
garbage truck 0.05656484141945839
French horn 0.000811699777841568
English springer 0.0005154424579814076
gas pump 0.00035581536940298975
--------------------------------
```

From `mycnn` model

```
--------------------------------
Detecting for class test-garbagetruck.jpg model mycnn
--------------------------------
garbage truck 0.9999910593032837
tench 5.209851224208251e-06
French horn 1.5582596688545891e-06
cassette player 1.4181257483869558e-06
chain saw 5.445947408588836e-07
--------------------------------
Detecting for class test-truck.jpg model mycnn
--------------------------------
garbage truck 0.9966506361961365
gas pump 0.0030325346160680056
parachute 0.00017644459148868918
golf ball 0.00013912079157307744
chain saw 8.742731552047189e-07
--------------------------------
```
Though accuracy of my model with validation images is very low and it is not able to detect the dog in the image

```
Detecting for class test-dog.jpg model mycnn
--------------------------------
gas pump 0.708825409412384
English springer 0.22835583984851837
chain saw 0.05654475837945938
church 0.0025733024813234806
golf ball 0.002355701057240367
--------------------------------
```
Next is to train with DropOut added to the model ; or some generalization in training like adding all animals and objects as super-set classes in the training itself


 Related repo: https://github.com/alexcpn/cnn_in_python

 
 