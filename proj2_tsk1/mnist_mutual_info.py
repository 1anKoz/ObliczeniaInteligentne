import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import fetch_openml
import pickle

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Convert string labels to integers
y = y.astype(np.uint8)

# Convert X to NumPy array
X = np.array(X)

# Feature extraction using mutual information
mutual_info = mutual_info_classif(X, y)

# Choose the number of features to extract
num_features_to_extract = 40  # Change this to the desired number of features

# Sort the features based on mutual information scores and select top N features
selected_features_indices = np.argsort(mutual_info)[-num_features_to_extract:]
selected_features = X[:, selected_features_indices]

# Save the selected features using pickle
with open('.\encodings\mnist_mutual_info_40.sav', 'wb') as f:
    pickle.dump(selected_features, f)