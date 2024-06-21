import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.colors import ListedColormap

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

def plot_decision_boundary(model, features, labels, title, filename):
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
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()



def eval_model(model_file, test_file, train_file):
    print("For " + model_file)
    model = pickle.load(open(model_file, 'rb'))
    test_features, test_labels = pickle.load(open(test_file, 'rb'))

    # Create DataLoader for features
    test_feature_loader = DataLoader(torch.utils.data.TensorDataset(test_features, test_labels), batch_size=1, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    predicted_table = []
    real_table = []
    with torch.no_grad():
        for data, target in test_feature_loader:
            real_table.append(target.cpu().numpy())
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predicted_table.append(predicted.cpu().numpy())
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print("Accuracy: " + str(correct/total))

    conf_mx = confusion_matrix(real_table, predicted_table)
    print(conf_mx)

    if "_2_" in files[i]:
        train_features, train_labels = pickle.load(open(train_file, 'rb'))
        test_title = "Decision Boundary and Data Points for test data of " + model_file[9:len(model_file)-3]
        train_title = "Decision Boundary and Data Points for train data of " + model_file[9:len(model_file)-3]
        test_file = test_title[0:len(test_title)] + ".png"
        train_file = train_title[0:len(test_title)] + ".png"
        plot_decision_boundary(model, train_features.numpy(), train_labels.numpy(), train_title, train_file)
        plot_decision_boundary(model, test_features.numpy(), test_labels.numpy(), test_title, test_file)

    


#files = [".\models\CIFAR_M_2_features.sav", ".\models\CIFAR_M_100_features.sav", ".\models\MNIST_M_2_features.sav", ".\models\MNIST_M_100_features.sav"]
#test_files = [".\encodings\marta_test_2_features.sav", ".\encodings\marta_test_100_features.sav", ".\encodings\marta_MNIST_test_2_features.sav", ".\encodings\marta_MNIST_test_100_features.sav"]
#train_files = [".\encodings\marta_train_2_features.sav", ".\encodings\marta_train_100_features.sav", ".\encodings\marta_MNIST_train_2_features.sav", ".\encodings\marta_MNIST_train_100_features.sav"]

files = [".\models\MNIST_M_100_features_10.sav", ".\models\MNIST_M_100_features_20.sav", ".\models\MNIST_M_100_features_100.sav"]
train_files = [".\encodings\marta_MNIST_train_100_features_from_10.sav", ".\encodings\marta_MNIST_train_100_features_from_20.sav", ".\encodings\marta_MNIST_train_100_features_from_100.sav"]
test_files = [".\encodings\marta_MNIST_test_100_features.sav", ".\encodings\marta_MNIST_test_100_features.sav", ".\encodings\marta_MNIST_test_100_features.sav"]
for i in range(len(files)):
    eval_model(files[i], test_files[i], train_files[i])

    
        


