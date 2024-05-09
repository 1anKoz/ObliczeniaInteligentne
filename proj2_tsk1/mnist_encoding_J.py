import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from keras.datasets import mnist
import numpy as np
import cv2

#compactness, circularity, solidity, edge density, central, horizontal, vertical symmetries, extent


def compute_contour(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, _

# Extent represents the ratio of the object's area to the area of its bounding box.
# It provides information about how much of the bounding box is filled by the object.
def extent(image):
    contours, _ = compute_contour(image)
    
    largest_contour_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_contour_area:
            largest_contour_area = area
            largest_contour = contour
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    bounding_box_area = w * h
    extent = largest_contour_area / bounding_box_area
    
    return int(extent*1000)

# Circularity measures how closely an object resembles a perfect circle.
# It can be calculated as the ratio of the object's area to the square of its perimeter.
def circularity(image):

    contours, _ = compute_contour(image)
    
    largest_contour_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_contour_area:
            largest_contour_area = area
            largest_contour = contour
    
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = (4 * np.pi * largest_contour_area) / (perimeter ** 2)
    
    return int(circularity*1000)

# Compactness quantifies how compact or spread out an object is. It is calculated
# as the ratio of the object's area to the area of a circle with the same perimeter.
def compactness(image):

    contours, _ = compute_contour(image)

    largest_contour_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_contour_area:
            largest_contour_area = area
            largest_contour = contour
    
    perimeter = cv2.arcLength(largest_contour, True)
    compactness = largest_contour_area / (perimeter ** 2)
    
    return int(compactness*1000)

# Solidity measures the convexity of an object. It is the ratio of the object's area to the area of its convex hull.
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
    result = np.corrcoef(left_half.flatten(), np.flip(right_half, axis=1).flatten())[0, 1]
    return int(result*1000)

def vertical_symmetry_score(image):
    height, width = image.shape
    top_half = image[:height // 2, :]
    bottom_half = image[height // 2:, :]
    result = np.corrcoef(top_half.flatten(), np.flip(bottom_half, axis=0).flatten())[0, 1]
    return int(result*1000)

def central_symmetry_score(image):
    result = int(((horizontal_symmetry_score(image) + vertical_symmetry_score(image)) / 2))
    return result


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    vector_2 = []
    vector_8 = []
    vector_784 = []

    ctr = 0
for x in X_train:
    vector_2.append([central_symmetry_score(x), edge_density(x)])
    tmp = [central_symmetry_score(x), edge_density(x), horizontal_symmetry_score(x), vertical_symmetry_score(x), solidity(x), circularity(x), compactness(x), extent(x)]
    vector_8.append(tmp)
    ctr = ctr + 1
    if(ctr == 10): break

print(vector_2)
print()
print(vector_8)