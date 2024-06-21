import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle

dataset = ".\dat\CIFAR_subset_train_100.sav"
# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
#train_dataset = pickle.load(open(dataset, 'rb'))
train_dataset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model for feature extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 25, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=1),
            nn.Conv2d(25, 50, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=1),
            nn.Conv2d(50, 50, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=1)
        )
        self.fc = nn.Linear(9800, 2)

        # Change to desired feature size if needed (100 in this example)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

cnn_model = CNNFeatureExtractor()

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

train_features, train_labels = extract_features(cnn_model, train_loader)
test_features, test_labels = extract_features(cnn_model, test_loader)

# Save extracted features and labels using pickle
with open('./encodings/marta_train_2_features.sav', 'wb') as f:
    pickle.dump((train_features, train_labels), f)

with open('./encodings/marta_test_2_features.sav', 'wb') as f:
    pickle.dump((test_features, test_labels), f)
