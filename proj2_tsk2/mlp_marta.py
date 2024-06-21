import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filename = "MNIST_M_100_features_100"
# Load extracted features and labels using pickle
with open('./encodings/marta_MNIST_train_100_features_from_100.sav', 'rb') as f:
    train_features, train_labels = pickle.load(f)

with open('./encodings/marta_MNIST_test_100_features.sav', 'rb') as f:
    test_features, test_labels = pickle.load(f)

# Create DataLoader for features
train_feature_loader = DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), batch_size=64, shuffle=True)
test_feature_loader = DataLoader(torch.utils.data.TensorDataset(test_features, test_labels), batch_size=64, shuffle=False)

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

mlp_model = MLPClassifier(100, 10)  # Adjust input_dim if needed (100 in this example)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Train the MLP model
def train(model, loader, criterion, optimizer, epochs=1000):
    model.train()
    best_model = model
    best_accuracy = 0
    test_accuracy_table = []
    train_accuracy_table = []
    plot_exes = []
    i = 1
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        curr_acc = evaluate(model, False)
        train_accuracy_table.append(curr_acc)
        curr_acc = evaluate(model, True)
        test_accuracy_table.append(curr_acc)
        plot_exes.append(i)
        i += 1
        if curr_acc > best_accuracy:
            best_accuracy = curr_acc
            best_model = model
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
    plt.plot(plot_exes, test_accuracy_table, label="Test accuracy")
    plt.plot(plot_exes, train_accuracy_table, label="Train accuracy")
    title = "Accuracy of model for " + filename + " across epochs"
    plt.title(title)
    plt.legend(loc="lower right")
    saving = filename + ".png"
    plt.xlabel ('epoch')
    plt.ylabel ('accuracy')
    plt.savefig(saving)
    plt.clf()
    plt.cla()
    plt.close()
    return best_model

best_model = train(mlp_model, train_feature_loader, criterion, optimizer)
for_save = ".\models\\" + filename + ".sav"
pickle.dump(best_model, open(for_save, 'wb'))

    


