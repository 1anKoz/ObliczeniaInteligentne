import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.colors import ListedColormap
import os

# Load extracted features and labels using pickle
with open('./encodings/train_100_features.sav', 'rb') as f:
    train_features, train_labels = pickle.load(f)

with open('./encodings/test_100_features.sav', 'rb') as f:
    test_features, test_labels = pickle.load(f)

# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader for features
train_feature_loader = DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), batch_size=64, shuffle=True)
test_feature_loader = DataLoader(torch.utils.data.TensorDataset(test_features, test_labels), batch_size=64, shuffle=False)

# Define the MLP model for classification
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

mlp_model = MLPClassifier(100, 10)  # Adjust input_dim if needed (100 in this example)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Train the MLP model
def train(model, train_loader, test_loader, criterion, optimizer, epochs=100):  # Reduced epochs for demonstration
    model.train()
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    
    return train_accuracies, test_accuracies

# train_accuracies, test_accuracies = train(mlp_model, train_feature_loader, test_feature_loader, criterion, optimizer)

# Save the trained model using pickle
models_dir = './models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model_path = os.path.join(models_dir, 'cifar10_model')
with open(model_path, 'wb') as f:
    pickle.dump(mlp_model, f)
print(f"Model saved to {model_path}")