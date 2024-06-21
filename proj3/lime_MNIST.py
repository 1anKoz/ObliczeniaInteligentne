import torch
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and add channel dimension
X_test_tensor = torch.tensor(X_test / 255.0, dtype=torch.float32).unsqueeze(1)

# Load the pre-trained model
model = pickle.load(open(".\\models\\mdl_train_our", 'rb'))

# Define the prediction function for LIME
def predict(images):
    model.eval()
    # Convert images to the required shape and type for the model
    if images.ndim == 4 and images.shape[-1] == 3:  # If LIME passes 3-channel images
        images = images[..., 0]  # Use only the first channel
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    with torch.no_grad():
        outputs = model(images)
    # Reshape the outputs appropriately
    outputs = outputs.view(outputs.size(0), -1)
    # Convert the outputs to probabilities
    probabilities = F.softmax(outputs, dim=1).numpy()
    return probabilities

# Initialize the LimeImageExplainer
explainer = lime_image.LimeImageExplainer()

# Choose a sample to analyze
sample_index = 1
X_sample = X_test[sample_index]
y_sample = y_test[sample_index]

# Explain the prediction for the chosen sample
explanation = explainer.explain_instance(
    X_sample, 
    predict, 
    top_labels=1, 
    hide_color=0, 
    num_samples=1000
)

# Get the image and mask for the top class
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)

from skimage.color import gray2rgb, rgb2gray, label2rgb # since the code wants color images
# label = y_sample
# print(mask)
# plt.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
# plt.show()

# Plot the original image and the LIME explanation
fig, ax = plt.subplots(1, 2)
ax[0].imshow(X_sample, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(mark_boundaries(temp, mask))
ax[1].set_title('LIME Explanation')
plt.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
plt.show()