# MLP Neural Network trained for the MNIST 0-9 integer Data Set
# Author = Adam Loeckle
# Date = 10/1/2021
# Class = ECEN4303

# - Numpy write ability warning, tests pass regardless of this warning

import torch
from torch.nn.modules import loss
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from fxpmath import Fxp

from utils import createTestImage, toHex

import numpy as np
import struct

# Train on CPU
# CUDA fails, PyTorch 1.6 3000 series compatability issue, not a large enough training set to bother fixing
device = torch.device("cpu")

# Path for model save/load and usable verilog weights
path = "weights.pth"
hidden_weights_path = "hidden_weights.txt"
hidden_biases_path = "hidden_biases.txt"
output_weights_path = "output_weights.txt"
output_biases_path = "output_biases.txt"


# MNIST Dataset and loaders
train_data = datasets.MNIST(root = 'data', train = True, transform = ToTensor(), download = True)
test_data = datasets.MNIST(root = 'data', train = False, transform = ToTensor())

loaders = {
    'train' : DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0),
    
    'test'  : DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0),
}

# MLP Neural network class
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # Input size: 784 (28*28)
        # Output size: 120
        self.fc1 = nn.Linear(28*28, 120)

        # Input size: 120
        # Output size: 10
        self.fc2 = nn.Linear(120, 10)

        # Cross Entropy Loss Function
        self.loss_func = nn.CrossEntropyLoss()

        # Adam Optimizer
        # Learning Rate = 0.01
        self.optimizer = optim.Adam(self.parameters(), lr = 0.01)

        # Sends CNN to specified device (cuda or cpu)
        self.to(device)

    def forward(self, input):
        
        # Flattens input to 1*748
        x = input.view(-1, 28*28)

        # ReLU activation function, max(0, x)
        x = F.relu(self.fc1(x))
        
        # Output
        output = self.fc2(x)
        return output, x 


    # Saver and loader for pth weight files for evaluation/testing
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    # Saver for model weights to be used in the Verilog implimentation
    # Downloads to txt file
    def viewable_save(self, w_file, b_file):
        weights_file = open(w_file, "w")
        biases_file = open(b_file, "w")
        weights = np.ndarray.flatten(self.fc1.weight.detach().numpy())
        biases = np.ndarray.flatten(self.fc1.bias.detach().numpy())

        # Weights smaller in decimal than 0.001 will be obscure in the hex conversion
        # This checks and rounds the weights up to +-0.001
        for i in weights:
            if abs(i) < 0.001:
                if i < 0:
                    i = -0.001
                else:
                    i = 0.001

            # Using the fxpmath and toHex methods to turn the decimal floats into
            # 32 bit hexadecimal fixed point values
            final = toHex(Fxp(i, n_int=16, n_frac=16, signed=True).raw(), 32)
            
            # Cleans up the output, adds 0's to the beginning if it is not a full
            # 8 hex digits in length
            final = str(final).replace("0x", "")
            if (len(final) < 8):
                for i in range(8 - len(final)):
                    final = "0" + str(final)
            weights_file.write("{}\n".format(final))
            
        for i in biases:
            if abs(i) < 0.001:
                if i < 0:
                    i = -0.001
                else:
                    i = 0.001
            final = toHex(Fxp(i, n_int=16, n_frac=16, signed=True).raw(), 32)
            final = str(final).replace("0x", "")
            if (len(final) < 8):
                for i in range(8 - len(final)):
                    final = "0" + str(final)
            biases_file.write("{}\n".format(final))
        
        weights_file.close()
        biases_file.close()

# Training method for model creation
# - Tracks loss and displays the values to the screen for each epoch/step
def train(epochs, cnn, loader):
    cnn.train(mode=True)
    steps = len(loader['train'])

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(loader['train']):
            
            b_x = Variable(images)
            b_y = Variable(labels)

            output = cnn(b_x)[0]
            loss = cnn.loss_func(output, b_y)
            
            cnn.optimizer.zero_grad()
            loss.backward()
            cnn.optimizer.step()

            if (i + 1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, steps, loss.item()))

# Testing method for testing a saved model
# Tracks and displays accuracy of the inserted model
def test(cnn):
    cnn.train(mode=False)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            acc = (pred_y == labels).sum().item() / float(labels.size(0))
    
    print('Test Accuracy of the model on the 10000 test images: %.2f' % acc)

# Define the CNN for training and testing
if __name__ == "__main__":
    
    # CNN declaration
    mlp = MLP()

    # Train and save
    train(1, mlp, loaders)
    # cnn.save(path)

    # Load and test
    # cnn.load(path)
    test(mlp) 

    # Verilog text file save
    mlp.viewable_save(hidden_weights_path, hidden_biases_path)
    mlp.viewable_save(output_weights_path, output_biases_path)

    createTestImage(loaders['test'])