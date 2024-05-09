import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from keras.datasets import mnist
import numpy as np
import cv2


def edge_density(image):
    # Convert the image to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)  # Adjust thresholds as needed
    
    # Calculate edge density
    height, width = edges.shape
    total_pixels = height * width
    edge_pixels = np.count_nonzero(edges)
    density = edge_pixels / total_pixels
    
    return int(density*1000)

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
    result = int(((horizontal_symmetry_score(image) + vertical_symmetry_score(image)) / 2)*1000)
    return result


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    ctr = 0
for x in X_train:
    # print(int(horizontal_symmetry_score(x)*1000))
    # print("**********************************")
    # print(int(vertical_symmetry_score(x)*1000))
    # print("----------------------------------")
    print(central_symmetry_score(x))
    #print(edge_density(x))
    print()
    ctr = ctr + 1
    if(ctr == 10): break