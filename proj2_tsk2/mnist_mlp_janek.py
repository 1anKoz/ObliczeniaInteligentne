from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

# Load extracted features and labels using pickle
with open('./encodings/train_features_100_mnist.sav', 'rb') as f:
    train_features, train_labels = pickle.load(f)

with open('./encodings/test_features_100_mnist.sav', 'rb') as f:
    test_features, test_labels = pickle.load(f)

# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale images
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

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

mlp_model = MLPClassifier(100, 10)  # Input dimension is 128
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Train the MLP model
def train(model, train_loader, test_loader, criterion, optimizer, epochs=1000):  # Reduced epochs for demonstration
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

train_accuracies, test_accuracies = train(mlp_model, train_feature_loader, test_feature_loader, criterion, optimizer)

best_model = train(mlp_model, train_feature_loader, test_feature_loader, criterion, optimizer)
for_save = ".\models\\mnist_cnn_model"
pickle.dump(best_model, open(for_save, 'wb'))

# Plot training and test accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy Over Epochs for MNIST 2 ftrs')
plt.legend()
plt.show()

# # Evaluate the MLP model and plot confusion matrix
# def evaluate(model, loader, criterion):
#     model.eval()
#     correct = 0
#     total = 0
#     all_targets = []
#     all_predictions = []
#     with torch.no_grad():
#         for data, target in loader:
#             output = model(data)
#             _, predicted = torch.max(output.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()
#             all_targets.extend(target.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())
    
#     accuracy = 100 * correct / total
#     print(f'Accuracy: {accuracy:.2f}%')

#     # Compute confusion matrix
#     cm = confusion_matrix(all_targets, all_predictions)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix for MNIST 2 ftrs')
#     plt.show()

# evaluate(mlp_model, test_feature_loader, criterion)

# # Plot decision boundary
# def plot_decision_boundary(model, features, labels):
#     x_min, x_max = features[:, 0].min(), features[:, 0].max()
#     y_min, y_max = features[:, 1].min(), features[:, 1].max()
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                          np.arange(y_min, y_max, 0.01))
    
#     Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).detach().numpy()
#     Z = np.argmax(Z, axis=1)
#     Z = Z.reshape(xx.shape)
    
#     plt.figure(figsize=(10, 8))
#     plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF', '#FFAFAF', '#AFAFFF', '#AFAAAF', '#FFFFAA', '#FFAAFF', '#AAF1AF']))
#     scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, s=20, edgecolor='k', cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF', '#FFAFAF', '#AFAFFF', '#AFAAAF', '#FFFF00', '#FF00FF', '#00FFFF']))
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('Decision Boundary and Data Points for MNIST 2 ftrs')
#     plt.show()

# plot_decision_boundary(mlp_model, train_features.numpy(), train_labels.numpy())