import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()

        # define all layers, here
        # these are all the layers the model can have
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, output_dim)
        self.dropout  = nn.Dropout(0.2)
        self.sig      = nn.Sigmoid()
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
        # this specifies the order of the layers
        x = F.relu(self.fc1(x)) # add relu activation function for first layer
        x = self.dropout(x)     # add dropout layer
        x = self.fc2(x)         # add second layer w/o activation
        x = self.sig(x)         # add sigmoid as activation layer
        
        return x