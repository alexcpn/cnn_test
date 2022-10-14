# My CNN Model
# https://alexcpn.medium.com/cnn-from-scratch-b97057d8cef4
# To get the filter use this https://docs.google.com/spreadsheets/d/1tsi4Yl2TwrPg5Ter8P_G30tFLSGQ1i29jqFagxNFa4A/edit?usp=sharing

import torch.nn as nn
import logging as log

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)


class MyCNN2(nn.Module):


    def make_block(self,input_depth,out_channel,kernel_size,stride):
        cnn1 =  nn.Conv2d(in_channels=input_depth,out_channels=int(out_channel/2),kernel_size=kernel_size,stride=stride)
        bn1 = nn.BatchNorm2d(int(out_channel/2))
        relu1 =  nn.ReLU()
        cnn2 =  nn.Conv2d(in_channels=int(out_channel/2),out_channels=out_channel,kernel_size=kernel_size,stride=stride)
        bn2 = nn.BatchNorm2d(out_channel)
        relu2 =  nn.ReLU()
        nn_stack = nn.Sequential(cnn1,bn1,relu1,cnn2,bn2,relu2)
        return nn_stack

    def __init__(self):
        self.output_cnn = 19095# 178084 
        super(MyCNN2, self).__init__()
        self.input_width = 227
        self.input_height = 227
        self.in_channel =3
        out_channel =6
        kernel_size =5
        stride =2

        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        
        self.cnn_stack1 = self.make_block(self.in_channel,out_channel,kernel_size,stride)
        out = self.get_output(self.input_width,self.input_height,stride,kernel_size,out_channel/2)
        out = self.get_output(out[0],out[1],stride,kernel_size,out_channel)
        self.linear_stack = nn.Linear(out[0]*out[1]*out[2],1000)
        print("Linear 1 out ",out[0]*out[1]*out[2])

        self.cnn_stack2 = self.make_block(self.in_channel,out_channel*2,kernel_size,stride)
        out = self.get_output(self.input_width,self.input_height,stride,kernel_size,out_channel)
        out = self.get_output(out[0],out[1],stride,kernel_size,out_channel*2)
        self.linear_stack2 = nn.Linear(out[0]*out[1]*out[2],1000)
        print("Linear 2 out ",out[0]*out[1]*out[2])

        self.cnn_stack3 = self.make_block(self.in_channel,out_channel*4,kernel_size,stride)
        out = self.get_output(self.input_width,self.input_height,stride,kernel_size,out_channel*2)
        out = self.get_output(out[0],out[1],stride,kernel_size,out_channel*4)
        self.linear_stack3 = nn.Linear(out[0]*out[1]*out[2],1000)
        print("Linear 3 out ",out[0]*out[1]*out[2])

            

        self.linear_stack4 = nn.Linear(1000,1000)
        self.linear_stack5 = nn.Linear(1000,10)

    def forward(self, x):
        logits1 = self.cnn_stack1(x)
        logits1 = self.flatten(logits1)
        logits1 = self.linear_stack(logits1)
        
        logits2 = self.cnn_stack2(x)
        logits2 = self.flatten(logits2)
        logits2 = self.linear_stack2(logits2)

        logits3 = self.cnn_stack3(x)
        logits3 = self.flatten(logits3)
        logits3 = self.linear_stack3(logits3)
       
        logits4 =self.linear_stack4(logits3)
        logits5 =self.linear_stack5(logits4)
        
        return logits5

    def get_output(self,input_width:int,input_height:int,stride:int,kernel_size:int,out_channels:int,padding=0)-> tuple[int,int,int]:
        """
        get the output_width,output_height and output_depth after a convolution of input 
        param input_width:  input_width
        param input_height:  input_height
        param stride:  stride of the CNN
        param kernel_size:  kernel_size or filter size of the CNN
        param out_channels:  out_channels specified in the CNN layer
        param padding:  padding defaults to 0
        returns (output_width,output_height,output_depth)
        """
        # Formula (W) = (W-F + 2P)/S +1
        # Formula (H) = (H-F + 2P)/S +1
        output_width = (input_width - kernel_size + 2*padding)/stride +1
        output_height = (input_height - kernel_size + 2*padding)/stride +1
        output_depth = out_channels
        return (int(output_width),int(output_height),output_depth)
    