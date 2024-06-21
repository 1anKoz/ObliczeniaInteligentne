import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
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
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
X_test = X_test / 255.0

# Load the pre-trained model (ensure this model is trained on CIFAR-10)
model_path = '/content/drive/MyDrive/mdl_MNIST_CNN.sav'
with open(model_path, 'rb') as f:
    model = pickle.load(f)
model.eval()
explainer = lime_image.LimeImageExplainer()

# Define the CNN model for feature extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_size):
        super(CNNFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(576, output_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

class ModelWrapper():
    def __init__(self, model, cnn):
        self.mdl = model
        self.cnn = cnn

    def my_predict(self, image, rest=None):
        if rest == None:
            return self.mdl((self.cnn(image)))
        else:
            return self.mdl((self.cnn(image)), rest)



# Define the prediction function for LIME
def predict(images):
    model.eval()
    images = images[:, :, :]
    new_img = []
    for b in range(len(images)):
      new_img.append([[]])
      for h in range(len(images[0])):
        new_img[b][0].append([])
        for w in range(len(images[0][0])):
          new_img[b][0][h].append(images[b][h][w][0])

    images = new_img
    # Convert images to the required shape and type for the model
    images = torch.tensor(images, dtype=torch.float32) #.permute(0, 3, 1, 2)  # Change shape to (batch_size, channels, height, width)
    batch_size = images.shape[0]

    cnn_model = CNNFeatureExtractor(100)
    wrap = ModelWrapper(model, cnn_model)

    with torch.no_grad():
        outputs = wrap.my_predict(images)
    # Convert the outputs to probabilities
    probabilities = F.softmax(outputs, dim=1).numpy()
    return probabilities

# Choose a sample to analyze
sample_index = 3
#6 10 13
X_sample = X_test[sample_index]
y_sample = y_test[sample_index]
print(y_sample)
plt.imshow(X_sample, cmap='binary')
plt.show()

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
label = y_sample
print(mask)
plt.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
plt.show()


# Plot the original image and the LIME explanation
fig, ax = plt.subplots(1, 2)
ax[0].imshow(X_sample, cmap='binary')
ax[0].set_title('Original Image')
ax[1].imshow(mark_boundaries(temp, mask))
ax[1].set_title('LIME Explanation')
plt.show()
