import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch.nn as nn
import pickle

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

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the data
X_test = X_test / 255.0

# Load the pre-trained model (ensure this model is trained on CIFAR-10)
model_path = './models/cifar10_model'
with open(model_path, 'rb') as f:
    model = pickle.load(f)
model.eval()

# Define the prediction function for LIME
def predict(images):
    model.eval()
    # Convert images to the required shape and type for the model
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # Change shape to (batch_size, channels, height, width)
    batch_size = images.shape[0]
    # Flatten each image from (3, 32, 32) to (3072,)
    flattened_images = images.reshape(batch_size, -1)
    # Select the first 100 features for the model
    selected_features = flattened_images[:, :100]
    with torch.no_grad():
        outputs = model(selected_features)
    # Convert the outputs to probabilities
    probabilities = F.softmax(outputs, dim=1).numpy()
    return probabilities

# Initialize the LimeImageExplainer
explainer = lime_image.LimeImageExplainer()

# Choose a sample to analyze
sample_index = 5
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
ax[0].imshow(X_sample, cmap='binary')
ax[0].set_title('Original Image')
ax[1].imshow(mark_boundaries(temp, mask))
ax[1].set_title('LIME Explanation')
plt.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
plt.show()