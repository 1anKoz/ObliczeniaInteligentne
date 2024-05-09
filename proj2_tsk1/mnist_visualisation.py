import matplotlib.pyplot as plt
import tensorflow
from keras.datasets import mnist
import numpy as np

# # Load the MNIST dataset
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# # Plot the first 25 images in the training set
# plt.figure(figsize=(1, 10))
# for i in range(3845, 3855):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(train_images[i], cmap='gray')
#     plt.title(f"Label: {train_labels[i]}")
#     plt.axis('off')
# plt.show()

# vector = []

# for x in range(10):
#     vector.append([str(x), "ccc"])

# print(vector)

# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# ctr = 0
# X_train = X_train.reshape(len(X_train), 784)
# for x in X_train:
#     print(x)
#     ctr = ctr + 1
#     if(ctr == 5): break
# print(X_train)
# print()
# print(len(X_train))
# print()
# print(X_train.reshape(len(X_train), 784))

import matplotlib.pyplot as plt
from keras.datasets import mnist  # Assuming you have Keras installed

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select the range of images
start_index = 3845
end_index = 3855
selected_images = x_test[start_index:end_index]

# Plot the images
plt.figure(figsize=(10, 5))
for i in range(len(selected_images)):
    plt.subplot(2, 5, i+1)  # 2 rows, 5 columns
    plt.imshow(selected_images[i], cmap='gray')
    plt.title(f"Image {start_index+i}")
    plt.axis('off')
plt.show()