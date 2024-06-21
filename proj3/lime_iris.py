import torch
import pickle
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer

# Load and split the data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Load the scaler and transform the test data
scaler = pickle.load(open('./encodings/iris_scaler.sav', 'rb'))
X_test_scaled = scaler.transform(X_test)

# Convert to tensor
X_test_tensor = torch.Tensor(X_test_scaled)

# Load the pre-trained model
model = pickle.load(open('./models/iris_mdl', 'rb'))

# Define a prediction function for LIME
def predict_proba(X):
    X_tensor = torch.Tensor(X)
    with torch.no_grad():
        outputs = model(X_tensor).numpy()
    return outputs

# Initialize LIME explainer
explainer = LimeTabularExplainer(X_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

# List to hold all LIME explanations
lime_data = []

# Iterate over each sample in the test set
for i in range(len(X_test)):
    X_sample = X_test_scaled[i:i+1]
    y_sample = y_test[i]
    
    # Explain the model's prediction using LIME
    explanation = explainer.explain_instance(X_sample.flatten(), predict_proba, num_features=4, top_labels=1)
    
    # Get the explanation for the predicted class
    label = explanation.top_labels[0]
    exp = explanation.as_list(label=label)
    
    # Extract and format the explanation data
    features, values = zip(*exp)
    lime_entry = list(values) + [label, y_sample]
    lime_data.append(lime_entry)

# Define the header for the CSV file
header = iris.feature_names + ['Predicted Class', 'Actual Class']

# Write the data to a CSV file
csv_file_path = './lime_data.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(header)
    writer.writerows(lime_data)

print(f"LIME data saved to {csv_file_path}")