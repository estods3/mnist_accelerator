#
# Copyright (c) 2024 Evan Stoddart
# github.com/estods3
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
import torch.quantization as quant
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

from data_preprocessor import preprocessor, generate_cocotb_tests

# Define a custom observer for 2-bit quantization
#
#
#
class TwoBitQuantizationObserver(quant.ObserverBase):
    def __init__(self, dtype=torch.qint8, quant_min=-2, quant_max=1):
        super().__init__(dtype=dtype)
        self.dtype = dtype
        self.quant_min = quant_min
        self.quant_max = quant_max

    def calculate_qparams(self):
        scale = 1.0
        zero_point = 0
        return scale, zero_point

    def forward(self, x):
        # Quantize to 2 bits
        return torch.clamp(x.round(), self.quant_min, self.quant_max)

######################################################################
# Neural Network Model                                               #
# --------------------                                               #
# Desc: pytorch neural network design used to classify MNIST images  #
# NOTE: The neural network defined here will be used to train, test, #
# and deploy to verilog for Tiny Tapeout Design.                     #
#                                                                    #
######################################################################
# MNIST Pytorch Example:
#
#class MNISTEXAMPLENet(nn.Module):
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  #1,32,3,1
        self.conv2 = nn.Conv2d(16, 32, 3, 1) #32,64,3,1
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(800, 128) #9216
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Claude.ai
class ClaudeNet(nn.Module):
#class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input layer: 196 inputs (14x14 flattened image)
        # First hidden layer: 128 neurons with ReLU activation
        self.fc1 = nn.Linear(196, 128)
        # Second hidden layer: 64 neurons with ReLU activation
        self.fc2 = nn.Linear(128, 64)
        # Output layer: 10 neurons (one for each digit 0-9)
        self.fc3 = nn.Linear(64, 10)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten the input if it's not already flattened
        # TODO - why is flattening needed for this Net() and not the other???
        if len(x.shape) > 2:
            print("FLATTENING")
            x = x.view(x.size(0), -1)
        
        # Pass through first hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Pass through second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation here; will be applied in loss function)
        x = self.fc3(x)
        return x


# Training Helper Function
# desc: perfom training on a model given the parameters
# inputs:
# returns: None
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Logging, Training Visualization
    train_losses = []
    test_losses = []
    test_accuracies = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


        #TODO - keep track of training progression. create plots that display and save to files to be embedded in README
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        #test_loss = test_loss / len(test_loader)
        #test_losses.append(test_loss)
        
        #accuracy = 100 * correct / total
        #test_accuracies.append(accuracy)

# Testing Helper Function
# desc: test the model on a test set
# inputs:
# returns: none
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Train and Save Model
# desc: sets up model, training/test sets, etc. and calls train()
# Inputs: None
# Returns: model, trained
def train_model():
    # Training Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False, help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    t = preprocessor()
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=t)
    dataset2 = datasets.MNIST('../data', train=False, transform=t)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Example usage with quantization aware training
    #model.qconfig = quant.QConfig(
    #    activation=TwoBitQuantizationObserver,
    #    weight=TwoBitQuantizationObserver
    #)
    #model = quant.prepare(model)

    #model.qconfig = quant.get_default_qat_qconfig('fbgemm')   # 6% accuracy
    #model = quant.prepare_qat(model)

    # Train the model with quantization-aware training
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    
    #model = quant.convert(model)

    # FP16 inference
    #model = model.half()  # Convert model to FP16

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #for epoch in range(1, args.epochs + 1):
    #    #train(args, model, device, train_loader, optimizer, epoch)
    #    test(model, device, test_loader)
    #    scheduler.step()
    
    return model

# Test Model
# desc: sets up model, gets test set, etc. and calls test()
# Inputs: model - trained pytorch net()
# Returns: None
def test_model(model):
    # Testing Settings
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    test_kwargs = {'batch_size': 1000}
    torch.manual_seed(1)

    # Configuring Hardware Device
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        test_kwargs.update(cuda_kwargs)

    # Gather Test Set
    t = preprocessor()
    dataset2 = datasets.MNIST('../data', train=False, transform=t)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Test
    test(model, device, test_loader)

    # Test Results/Analytics
    # TODO - create plots that display and save to files to be embedded in README
    # TODO - issues with matplotlib plotting. core dumped
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Convert Model to Verilog
# Desc: Convert Pytorch model into a hardware description language such as Verilog
# Inputs: None
# Returns: None
def convert_model_to_verilog():
    print("Converting model to verilog")
    # TODO

# Generate Test-Cases
# Desc: Generate cocotb test-cases based on images from the MNIST test set.
# Inputs: None
# Returns: None
def generate_testcases():
    save_images = False

    # Load Data
    # ---------
    test_kwargs = {'batch_size': 64}
    transform = preprocessor()
    testdataset = datasets.MNIST('../data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testdataset, **test_kwargs)

    # Save Preprocessed Images as dataframe
    # -------------------------------------
    print("Saving Test Images in batches: ")
    test_dataframe = pd.DataFrame(columns=["batch", "sample", "data vector", "label"])
    for batch_idx, (data, target) in enumerate(test_loader):
        print(str(batch_idx) + " ", end='')
        if(data.shape == (test_kwargs["batch_size"], 1, 14, 14)):
            for i, single_image in enumerate(data):
                flat_list = single_image.numpy().flatten(order='C').tolist()
                vector = ''.join(str(int(x)) for x in flat_list)
                row = {"batch":[batch_idx], "sample":[i], "data vector": [vector], "label":[int(target[i])]}
                new_row = pd.DataFrame(data=row)#,columns=["batch", "sample", "data vector", "label"])
                test_dataframe.reset_index(drop=True, inplace=True)
                new_row.reset_index(drop=True, inplace=True)
                test_dataframe = pd.concat([test_dataframe, new_row], ignore_index=True)
                if(save_images):
                    utils.save_image(single_image, '../data/MNIST/processed/test/batch{}_sample{}_class{}.png'.format(batch_idx, i, target[i]), normalize=False)
        if(batch_idx > 0):
            break

    # Generate Verilog Test Cases
    # ---------------------------
    print(test_dataframe.head())
    test_dataframe = test_dataframe.sample(n = 10)
    generate_cocotb_tests(test_dataframe, "randomtests.py")
    print("Cocotb testcases saved to randomtests.py")

if __name__ == '__main__':
    print("")
    print("Run from terminal without OSS CAD Suite enabled!")
    print("")
    print("MNIST Python Utility:")
    print("\t(1) Train Pytorch Model and save parameters")
    print("\t(2) Test Pytorch Model against MNIST testset")
    print("\t(3) Convert Pytorch Model to Verilog file neuralnetwork.v")
    print("\t(4) Generate cocotb testcases for Verilog testbench (test.py)")
    print("\t(5) Perform Benchmark - Run Pytorch and Tiny Tapeout chip in parallel (TBD)")

    selection = input("Enter an option (1-5): ")
    if(selection == '1'):
        print("Training Model...")
        model = train_model()
        print("Training Model...Done")
        print("Saving Model...")
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Saving Model...Done")
    elif(selection == '2'):
        print("Loading Model...")
        model = Net()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.eval()
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loading Model...Done")
        print("Testing Model...")
        test_model(model)
        print("Testing Model...Done")
    elif(selection == '3'):
        print("Loading Model")
        model = Net()
        model.load_state_dict(torch.load('mnist_cnn.pt'))
        model.eval()
        print("Loading Model...Done")
        print("Converting Model...")
        convert_model_to_verilog()
        print("Converting Model...Done")

    elif(selection == '4'):
        print("Generating Tests...")
        generate_testcases()
        print("Generating Tests...Done")
    else:
        print("Invalid Option...exiting.")
