import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from keras.datasets import mnist
import numpy as np
import cv2

def compute_contour(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, _

def solidity(image):
    contours, _ = compute_contour(image)
    
    largest_contour_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_contour_area:
            largest_contour_area = area
            largest_contour = contour
    
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = largest_contour_area / hull_area
    
    return int(solidity*1000)

#compactness, eccentricity, (circularity), (solidity)

def edge_density(image):
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
    vector_2 = []
    vector_7 = []
    vector_784 = []

    ctr = 0
for x in X_train:
    #vector_2.append([central_symmetry_score(x), edge_density(x)])
    print(solidity(x))
    print()
    ctr = ctr + 1
    if(ctr == 10): break