import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.transforms import v2
import numpy as np
from matplotlib.colors import ListedColormap
import os
from torch import Tensor
from captum.robust import MinParamPerturbation

# Define the MLP model for classification
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Define the CNN model for feature extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self, out):
        super(CNNFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=1),
            nn.Conv2d(25, 50, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=1),
            nn.Conv2d(50, 50, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=1)
        )
        self.fc = nn.Linear(5000, out)  # Change to desired feature size if needed (100 in this example)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

transforms_aug1 = transforms.Compose([
        transforms.AutoAugment(), #augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        
    ])

transforms_aug2 = transforms.Compose([
        transforms.RandomRotation(30), #augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        
    ])

transforms_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
def get_dataset(trans):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=trans)

    return test_dataset, test_dataset, test_dataset, test_dataset

# Evaluate the MLP model
def evaluate(model, test_data):
    if test_data:
        loader = DataLoader(torch.utils.data.TensorDataset(test_features, test_labels), batch_size=64, shuffle=False)
    else:
        loader = DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    model.train()
    return correct/total


# Train the MLP model
def train(model, loader, criterion, optimizer, epochs=1000):
    model.train()
    best_model = model
    best_accuracy = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        curr_acc = evaluate(model, True)
        if curr_acc > best_accuracy:
            best_accuracy = curr_acc
            best_model = model
            
        #print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
    return best_model, best_accuracy

# Extract features using the CNN
def extract_features(model, loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            features.append(output)
            labels.append(target)
    return torch.cat(features), torch.cat(labels)

def gaussian_noise(inp: Tensor, std: float) -> Tensor:
    return inp + std*torch.randn_like(inp)

from random import randint

def coverup(inp: Tensor, std: float) -> Tensor:
    new_inp = inp
    for i in range(std):
        new_inp[0, 0, randint(0, 27), randint(0, 27)] = 0
    return new_inp

def fill_in(inp: Tensor, std: float) -> Tensor:
    new_inp = inp
    for i in range(std):
        new_inp[0, 0, randint(0, 27), randint(0, 27)] = 255
    return new_inp

def brighten(inp: Tensor, std: float) -> Tensor:
    new_inp = inp
    for i in range(std):
        new_inp[0, 0, randint(0, 27), randint(0, 27)] += 10
        if new_inp[0, 0, randint(0, 27), randint(0, 27)] > 255:
            new_inp[0, 0, randint(0, 27), randint(0, 27)] = 255
    return new_inp

class ModelWrapper():
    def __init__(self, model, cnn):
        self.mdl = model
        self.cnn = cnn

    def my_predict(self, image, rest=None):
        if rest == None:
            return self.mdl((self.cnn(image)))
        else:
            return self.mdl((self.cnn(image)), rest)



import torchvision.transforms as T

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_features = [100]
for i in range(1):
    if True:
        train_dataset_table = [[], [], []]
        test_dataset = None
        if True:
            test_dataset, train_dataset_table[0], train_dataset_table[1], train_dataset_table[2] = get_dataset(transforms_no_aug)

            in_train_dataset_table = 0
            train_dataset = train_dataset_table[2]
            if True:
            #for train_dataset in train_dataset_table:
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                for feature_size in out_features:
                    cnn_model = CNNFeatureExtractor(feature_size)
                    best_model = pickle.load(open('./models/mdl.sav', 'rb'))
                    
                    
                    #for_save = ".\models\\" + filename + ".sav"
                    #pickle.dump(best_model, open(for_save, 'wb'))
                    
                    wrap = ModelWrapper(best_model, cnn_model)
                    min_pert = MinParamPerturbation(forward_func=wrap.my_predict,
                                   attack=fill_in,
                                   arg_name="std",
                                   arg_min=0,
                                   arg_max=1500,
                                   arg_step=1,
                                )
                    for images, labels in test_loader:
                        oryg_img = images[0][0].numpy().copy()
                        plt.imshow(oryg_img, cmap='binary')
                        plt.title("Oryginal image")
                        
                        
                        with torch.no_grad():
                            output = wrap.my_predict(images)
                            _, predicted = torch.max(output.data, 1)
                            if (predicted != int(labels[0])):
                                print("Incorrectly predicted " + str(int(labels[0])) + " as " + str(int(predicted[0])))
                                continue

                            noised_image, min_std = min_pert.evaluate(inputs=images, target=labels)
                            output2 = wrap.my_predict(noised_image)
                            _, predicted2 = torch.max(output2.data, 1)
                            if (predicted != predicted2):
                                print("Real label: " + str(int(labels[0])) + " No modifications label: " + str(int(predicted[0])) + " With modifications label: " + str(int(predicted2[0])) + " changed by: " + str(int(min_std)))
                                plt.show()
                                plt.imshow(T.ToPILImage()(noised_image[0]), cmap='binary')
                                plt.title("Modified image")
                                plt.show()
                            else:
                                print(str(int(labels[0])) + " predicted as " + str(int(predicted[0])))
                                


                        

              


