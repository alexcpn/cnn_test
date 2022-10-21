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

### How to Train

```
cnn_2$ /usr/bin/python3 /home/alex/coding/cnn_2/cnn/train_cnn.py
```
## Testing a saved model

Testing is happening in [cnn/test_cnn.py](cnn/test_cnn.py)

```
 /usr/bin/python3 /home/alex/coding/cnn_2/cnn/test_cnn.py
 ```

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
## Testing a saved with GradCam

Gradcam helps one visualize which parts of the images are important for the CNN when it classifies an object with high probability. After testing a model, you can use this to visualize and debug the test results

To Run

```
/usr/bin/python3 /home/alex/coding/cnn_2/cnn/gradcam_test.py
```

Output in gradcam_out folder

Example output for calssification of FrenchHorn by the ResNet50 model here 

![](https://i.imgur.com/vhxaB2d.png)



 Related repo: https://github.com/alexcpn/cnn_in_python

 
 