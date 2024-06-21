import torch
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms

# Define the MLP model class (same as in training script)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to compute saliency map
def compute_saliency(model, X, target_class):
    model.eval()
    X.requires_grad_()

    # Forward pass
    output = model(X)
    
    # Compute the loss based on the target class
    loss = -output[0, target_class].sum()  # Negative log likelihood for the target class
    
    # Backward pass to get gradients
    loss.backward()
    
    # Get the gradients of the input
    saliency = X.grad.data.abs().numpy()
    
    return saliency

# Load the model
model_path = './models/cifar10_model'
with open(model_path, 'rb') as f:
    mlp_model = pickle.load(f)
mlp_model.eval()

# Load the test features and labels
with open('./encodings/test_100_features.sav', 'rb') as f:
    test_features, test_labels = pickle.load(f)

# Convert test features to tensor
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Choose a sample to analyze
sample_index = 0
X_sample = test_features_tensor[sample_index:sample_index+1]
y_sample = test_labels[sample_index]

# Compute the saliency map
saliency = compute_saliency(mlp_model, X_sample, y_sample)

# Get the original image
original_image, original_label = cifar10_test[sample_index]
original_image = original_image.permute(1, 2, 0).numpy()

# Plot the original image and the saliency map side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot original image
ax[0].imshow(original_image)
ax[0].set_title('Original CIFAR-10 Image')
ax[0].axis('off')

# Plot saliency map
ax[1].bar(range(X_sample.shape[1]), saliency.squeeze(), color='r')
ax[1].set_title('Saliency Map')

plt.show()