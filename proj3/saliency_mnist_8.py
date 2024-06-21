import torch
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.decomposition import PCA

# Function to compute saliency map
def compute_saliency(model, X, target_class):
    model.eval()
    X.requires_grad_()

    try:
        # Forward pass
        output = model(X)
        print(f"Output shape: {output.shape}")

        # Ensure target_class is within the valid range
        assert target_class < output.size(1), "target_class index out of range"

        # Compute the loss based on the target class
        loss = -output[:, target_class].sum()  # Negative log likelihood for the target class

        # Backward pass to get gradients
        loss.backward()

        # Get the gradients of the input
        saliency = X.grad.data.abs().numpy()

        return saliency
    except Exception as e:
        print(f"Error during saliency computation: {e}")
        return None

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data and flatten the images
X_test = X_test / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Flatten to 784 dimensions

# Apply PCA to reduce dimensions to 8
pca = PCA(n_components=8)
X_test_pca = pca.fit_transform(X_test_flat)
X_test_tensor_pca = torch.tensor(X_test_pca, dtype=torch.float32)

# Load the second model and adjust input shape if necessary
model_path_second = './models/mnist_8_features'
with open(model_path_second, 'rb') as f:
    second_model = pickle.load(f)
second_model.eval()

# Choose a sample to analyze
sample_index = 1
X_sample_pca = X_test_tensor_pca[sample_index:sample_index+1]
y_sample = y_test[sample_index]

# Compute the saliency map for the second model
print("Computing saliency for the second model...")
saliency_second = compute_saliency(second_model, X_sample_pca, y_sample)
if saliency_second is not None:
    # Flatten the saliency map for the bar plot
    flattened_saliency_second = saliency_second.flatten()

# Feature labels
feature_labels = [
    "Central Symmetry Score", "Edge Density", "Horizontal Symmetry Score", 
    "Vertical Symmetry Score", "Solidity", "Circularity", "Compactness", "Extent"
]

# Plot the original image and the saliency bar diagram for the second model
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image
X_sample = X_test[sample_index]
ax[0].imshow(X_sample, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

if saliency_second is not None:
    # Plot saliency bar diagram for the second model
    ax[1].bar(range(flattened_saliency_second.size), flattened_saliency_second)
    ax[1].set_xticks(range(len(feature_labels)))
    ax[1].set_xticklabels(feature_labels, rotation=45, ha="right")
    ax[1].set_title('Saliency Map - Second Model')
    ax[1].set_xlabel('Feature')
    ax[1].set_ylabel('Saliency Value')

plt.tight_layout()
plt.show()