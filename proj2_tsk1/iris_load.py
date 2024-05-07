import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Step 2: Preprocess the data
# Convert labels to tensor
y = torch.tensor(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Create PyTorch dataset and dataloaders
class IrisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets and dataloaders
train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# with open('.\encodings\iris_train.sav', 'wb') as f_pkl:
#     pickle.dump(X_train, f_pkl)
# with open('.\encodings\iris_test.sav', 'wb') as f_pkl:
#     pickle.dump(X_test, f_pkl)

# with open('.\encodings\iris_train.sav', 'wb') as f_pkl:
#     pickle.dump(train_dataset, f_pkl)
# with open('.\encodings\iris_test.sav', 'wb') as f_pkl:
#     pickle.dump(test_dataset, f_pkl)