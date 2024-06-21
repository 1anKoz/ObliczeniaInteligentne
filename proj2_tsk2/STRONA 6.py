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
    
# Define the CNN model for feature extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self, out):
        super(CNNFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * 4 * 4, out)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

transforms_aug1 = transforms.Compose([
        transforms.AutoAugment(), #augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    ])

transforms_aug2 = transforms.Compose([
        transforms.RandomRotation(30), #augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    ])

transforms_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
def get_dataset(trans):
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)

    no_instances = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_no_instances = [100, 200, 1000]
    img_arr = [[], [], []]
    ys_arr = [[], [], []]
    for i in range(len(train_dataset)):
        class_int = int(train_dataset[i][1])
        for j in range(len(expected_no_instances)):
            if expected_no_instances[j] > no_instances[class_int]:
                img_arr[j].append(train_dataset[i])
                if (j == 2):
                    no_instances[class_int] += 1
            else:
                break
    return test_dataset, img_arr[0], img_arr[1], img_arr[2]

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

def plot_decision_boundary(model, features, labels, title):
    x_min, x_max = features[:, 0].min(), features[:, 0].max()
    y_min, y_max = features[:, 1].min(), features[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).detach().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF', '#FFAFAF', '#AFAFFF', '#AFAAAF', '#FFFFAA', '#FFAAFF', '#AAF1AF']))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, s=20, edgecolor='k', cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF', '#FFAFAF', '#AFAFFF', '#AFAAAF', '#FFFF00', '#FF00FF', '#00FFFF']))
    plt.title(title)
    filename = title + ".png"
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()

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

def fileoutput(line):
    file_output = open("Table.txt", "a")
    file_output.write(line)
    file_output.write("\n")
    file_output.close()
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_features = [2, 100]
fileoutput("Iteration;extraction method;set size;features used;accuracy")
print("Iteration;extraction method;set size;features used;accuracy")
for i in range(10):
    if True:
        train_dataset_table = [[], [], []]
        test_dataset = None
        for extr in range(3):
            if extr == 2:
                test_dataset, train_dataset_table[0], train_dataset_table[1], train_dataset_table[2] = get_dataset(transforms_no_aug)
            elif extr == 1:
                test_dataset, train_dataset_table[0], train_dataset_table[1], train_dataset_table[2] = get_dataset(transforms_aug1)
            else:
                test_dataset, train_dataset_table[0], train_dataset_table[1], train_dataset_table[2] = get_dataset(transforms_aug2)

            in_train_dataset_table = 0
            for train_dataset in train_dataset_table:
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

                for feature_size in out_features:
                    cnn_model = CNNFeatureExtractor(feature_size)

                    train_features, train_labels = extract_features(cnn_model, train_loader)
                    test_features, test_labels = extract_features(cnn_model, test_loader)

                    train_feature_loader = DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), batch_size=64, shuffle=True)
                    test_feature_loader = DataLoader(torch.utils.data.TensorDataset(test_features, test_labels), batch_size=64, shuffle=False)
                    mlp_model = MLPClassifier(feature_size, 10)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
                    best_model, accuracy = train(mlp_model, train_feature_loader, criterion, optimizer)
                    real_size = 100
                    if in_train_dataset_table == 1:
                        real_size = 200
                    if in_train_dataset_table == 2:
                        real_size = 1000
                    filename = "CIFAR_" + str(i) + "_extr_" + str(extr) + "_size_" + str(real_size) + "_features_" + str(feature_size)
                    lin = str(i) + ";" + str(extr) + ";" + str(real_size) + ";" + str(feature_size) + ";" + str(accuracy)
                    print(lin)
                    fileoutput(lin)
                    for_save = ".\models\\" + filename + ".sav"
                    pickle.dump(best_model, open(for_save, 'wb'))

                    for_db_train = "Train decision boundry " + filename
                    for_db_test = "Test decision boundry " + filename
                    if feature_size == 2:
                        plot_decision_boundary(best_model, train_features, train_labels, for_db_train)
                        plot_decision_boundary(best_model, test_features, test_labels, for_db_test)
                in_train_dataset_table += 1

os.system("shutdown /s /t 1") 
