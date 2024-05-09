import matplotlib.pyplot as plt
import tensorflow
from keras.datasets import mnist
import numpy as np

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Plot the first 25 images in the training set
plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')
plt.show()

# vector = []

# for x in range(10):
#     vector.append([str(x), "ccc"])

# print(vector)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

ctr = 0
X_train = X_train.reshape(len(X_train), 784)
for x in X_train:
    print(x)
    ctr = ctr + 1
    if(ctr == 5): break
# print(X_train)
# print()
# print(len(X_train))
# print()
# print(X_train.reshape(len(X_train), 784))