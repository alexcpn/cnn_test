# LeNet-5 Convolutional Neural Net in PyTorch

 Coding something similar to LeNet-5 Convolutional Neural Net in PyTorch
 
 
 The Network is as follows:
 
 ```
 Input (R,G,B)= [32.32.3] *(5.5.3)*6  == [28.28.6] * (5.5.6)*1 = [24.24.1] *  (5.5.1)*16 = [20.20.16] *
 FC layer 1 (20, 120, 16) == [20,120]* FC layer 2 (120, 1) == [20,1]* FC layer 3 (20, 10) == [10,1]* Softmax  (10,) =(10,1) = Output
```

 There may be slight changes in the Fully connected layer dimension as batching is involved

 Related repo: https://github.com/alexcpn/cnn_in_python

 