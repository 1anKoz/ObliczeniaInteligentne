import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from keras.datasets import mnist
import numpy as np


def horizontal_symmetry_score(image):
    height, width = image.shape
    left_half = image[:, :width // 2]
    right_half = image[:, width // 2:]
    return np.corrcoef(left_half.flatten(), np.flip(right_half, axis=1).flatten())[0, 1]

def vertical_symmetry_score(image):
    height, width = image.shape
    top_half = image[:height // 2, :]
    bottom_half = image[height // 2:, :]
    return np.corrcoef(top_half.flatten(), np.flip(bottom_half, axis=0).flatten())[0, 1]

def central_symmetry_score(image):
    return (horizontal_symmetry_score(image) + vertical_symmetry_score(image)) / 2


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    ctr = 0
for x in X_train:
    print(int(horizontal_symmetry_score(x)*1000))
    print("**********************************")
    print(int(vertical_symmetry_score(x)*1000))
    print("----------------------------------")
    print(int(central_symmetry_score(x)*1000))
    print()
    ctr = ctr + 1
    if(ctr == 10): break