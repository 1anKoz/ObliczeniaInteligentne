import torch
import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from lime.lime_tabular import LimeTabularExplainer

# Load and split the data
breast_cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2, random_state=42)

# Load the pre-trained model
model = pickle.load(open(".\\models\\dwbc_mdl", 'rb'))

# Convert to tensor
X_test_tensor = torch.Tensor(X_test)

# Define a prediction function for LIME
def predict_proba(X):
    X_tensor = torch.Tensor(X)
    with torch.no_grad():
        outputs = model(X_tensor).numpy()
    return outputs

# Initialize LIME explainer
explainer = LimeTabularExplainer(X_train, feature_names=breast_cancer.feature_names, class_names=breast_cancer.target_names, discretize_continuous=True)

# List to hold all LIME explanations
lime_data = []

# Iterate over each sample in the test set
for i in range(len(X_test)):
    X_sample = X_test[i]
    y_sample = y_test[i]
    
    # Explain the model's prediction using LIME
    explanation = explainer.explain_instance(X_sample, predict_proba, num_features=30, top_labels=1)
    
    # Get the explanation for the predicted class
    label = explanation.top_labels[0]
    exp = explanation.as_list(label=label)
    
    # Extract and format the explanation data
    lime_entry = [v for _, v in exp] + [label, y_sample]
    lime_data.append(lime_entry)

# Define the header for the CSV file
header = list(breast_cancer.feature_names) + ['Predicted Class', 'Actual Class']

# Write the data to a CSV file
csv_file_path = './lime_data_dwbc.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(header)
    writer.writerows(lime_data)

print(f"LIME data for Breast Cancer dataset saved to {csv_file_path}")