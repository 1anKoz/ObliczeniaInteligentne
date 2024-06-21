import torch
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Function to compute saliency map
def compute_saliency(model, X, target_class):
    model.eval()
    X.requires_grad_()

    # Forward pass
    output = model(X)
    
    # Compute the loss based on the target class
    loss = -output[:, :, 0, target_class].sum()  # Negative log likelihood for the target class
    
    # Backward pass to get gradients
    loss.backward()
    
    # Get the gradients of the input
    saliency = X.grad.data.abs().numpy()
    
    return saliency

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data and add a channel dimension
X_test = X_test / 255.0
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

# Load the model
model_path = './models/mdl_train_our'
with open(model_path, 'rb') as f:
    model = pickle.load(f)
model.eval()

# Choose a sample to analyze
sample_index = 1
X_sample = X_test_tensor[sample_index:sample_index+1]
y_sample = y_test[sample_index]

# Compute the saliency map
saliency = compute_saliency(model, X_sample, y_sample)

# Flatten the saliency map for the bar plot
flattened_saliency = saliency.flatten()

# Plot the original image and the saliency bar diagram
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image
ax[0].imshow(X_sample.detach().numpy().squeeze(), cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Plot saliency bar diagram
ax[1].bar(range(flattened_saliency.size), flattened_saliency)
ax[1].set_title('Saliency Map')
ax[1].set_xlabel('Pixel Index')
ax[1].set_ylabel('Saliency Value')

plt.show()