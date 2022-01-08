import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
from torch.distributions.normal import Normal
import sys

class SoftQNeuralNetworkTwin(nn.Module):
    def __init__(self,n_states,n_actions, h_dim):
        super(SoftQNeuralNetworkTwin,self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1\
                                     , out_channels=32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=32\
                                     , out_channels=64, kernel_size=4, stride=2)        
        self.conv3 = torch.nn.Conv2d(in_channels=64\
                                     , out_channels=64, kernel_size=3, stride=1)
        
        d_out = 11
        out_channels = 64
        #conv_output_dim = d_out*d_out*out_channels #with atari wrapper
        conv_output_dim = 28864
        
        #Q1
        self.linear11 = nn.Linear(conv_output_dim,h_dim)
        self.linear12 = nn.Linear(h_dim,n_actions)
        
        #Q2
        self.linear21 = nn.Linear(conv_output_dim,h_dim)
        self.linear22 = nn.Linear(h_dim,n_actions)

        self.activation = nn.ReLU()

    def forward(self,state):
        
        output_conf = self.conv1(state)
        output_conf = self.activation(output_conf)
        output_conf = self.conv2(output_conf)
        output_conf = self.activation(output_conf)
        output_conf = self.conv3(output_conf)
        output_conf = self.activation(output_conf)

        #flattening conv output tensor to put into feedforward
        output_conf = output_conf.reshape(output_conf.size(0),-1)
                
        output1 = self.linear11(output_conf)
        output1 = self.activation(output1)
        output1 = self.linear12(output1)

        output2 = self.linear21(output_conf)
        output2 = self.activation(output2)
        output2 = self.linear22(output2)

        return output1,output2

    
class SoftPiNeuralNetwork(nn.Module):
    
    def __init__(self,n_states,n_actions, h_dim, log_std_min=-20, log_std_max=2, init_w=3e-3, eps=0.0001):
        super(SoftPiNeuralNetwork,self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1\
                                     , out_channels=32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=32\
                                     , out_channels=64, kernel_size=4, stride=2)        
        self.conv3 = torch.nn.Conv2d(in_channels=64\
                                     , out_channels=64, kernel_size=3, stride=1)
        
        d_out = 11
        out_channels = 64
        #conv_output_dim = d_out*d_out*out_channels #with atari wrapper
        conv_output_dim = 28864        
        
        self.linear1 = nn.Linear(conv_output_dim,h_dim)
        self.linear2 = nn.Linear(h_dim,n_actions)
            
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(-1)
        
        self.n_actions = n_actions
        self.n_states = n_states   
        self.eps = torch.Tensor([eps]).cuda()
        

    def forward(self,state):
        
        output_conf = self.conv1(state)
        output_conf = self.activation(output_conf)
        output_conf = self.conv2(output_conf)
        output_conf = self.activation(output_conf)
        output_conf = self.conv3(output_conf)
        output_conf = self.activation(output_conf)
        
        output_conf = output_conf.reshape(output_conf.size(0),-1)
        
        output = self.linear1(output_conf)
        output = self.activation(output)
        output = self.linear2(output)
        
        output = output.clip(-10,10)

        return self.softmax(output)
    
    def get_prob(self, state):
        probs = self.forward(state) 
        return probs
  
    #evaluate action and log probability from state
    def sample(self, state, n_samples = 1):

        probs = self.forward(state)                   
        prob_dist = torch.distributions.Categorical(probs)
        action = prob_dist.sample(sample_shape=[n_samples])

        return action
    
