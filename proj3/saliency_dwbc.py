import torch
import torch.nn as nn
import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Function to compute saliency map
def compute_saliency(model, X, target_class):
    model.eval()
    X.requires_grad_()
    
    # Forward pass
    output = model(X)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Backward pass to get gradients
    output[:, target_class].backward()
    
    # Get the gradients of the input
    saliency = X.grad.data.abs().numpy()
    
    return saliency

# Load and split the data
breast_cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2, random_state=42)

# Load the scaler and transform the test data
scaler = pickle.load(open('./encodings/dwbc_scaler.sav', 'rb'))
X_test = scaler.transform(X_test)

# Convert to tensor
X_test_tensor = torch.Tensor(X_test)

# Load the pre-trained model
model = pickle.load(open('./models/dwbc_mdl', 'rb'))

# Iterate over each sample in the test set
all_data = []

for i in range(len(X_test)):
    X_sample = X_test_tensor[i:i+1]
    y_sample = y_test[i]
    
    # Compute the saliency map
    saliency = compute_saliency(model, X_sample, y_sample)
    
    # Compute the predicted class and probabilities
    with torch.no_grad():  # Disable gradient calculation for efficiency
        logits = model(X_sample)  # Forward pass to get model output (logits)
        probabilities = torch.softmax(logits, dim=1)  # Apply softmax to get probabilities
        predicted_class = probabilities.argmax(dim=1).item()  # Get the predicted class
    
    # Store the saliency, predicted class, and actual class
    data_entry = list(saliency[0]) + [predicted_class, y_sample.item()]
    all_data.append(data_entry)
    
    # Plot the saliency map for each sample (commented out)
    # plt.figure()
    # plt.bar(range(len(saliency[0])), saliency[0], tick_label=breast_cancer.feature_names)
    # plt.xlabel('Features')
    # plt.ylabel('Saliency')
    # plt.title(f'Saliency Map for Sample {i}')
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()

# Define the header for the CSV file
header = list(breast_cancer.feature_names) + ['Predicted Class', 'Actual Class']

# Write the data to a CSV file
csv_file_path = './saliency_data_dwbc.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(header)
    writer.writerows(all_data)

print(f"Saliency data saved to {csv_file_path}")