import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import pickle
import numpy as np
from iris_load import IrisDataset
from wine_load import WineDataset
from dwbc_load import DwbcDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":
    files = [".\encodings\J_enc_train_2_features.sav", ".\encodings\J_enc_train_8_features.sav"]
    files_test = [".\encodings\J_enc_test_2_features.sav", ".\encodings\J_enc_test_8_features.sav"]
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # files = [".\encodings\iris_train.sav"]
    # files_test = [".\encodings\iris_test.sav"]
    #X_train, X_test, Y_train, Y_test = train_test_split(load_iris().data, load_iris().target, test_size=0.2, random_state=42)
    
    # files = [".\encodings\wine_train.sav"]
    # files_test = [".\encodings\wine_test.sav"]
    # X_train, X_test, Y_train, Y_test = train_test_split(load_wine().data, load_wine().target, test_size=0.2, random_state=42)

    # files = [".\encodings\dwbc_train.sav"]
    # files_test = [".\encodings\dwbc_test.sav"]
    # X_train, X_test, Y_train, Y_test = train_test_split(load_breast_cancer().data, load_breast_cancer().target, test_size=0.2, random_state=42)

    
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 300
    for set_no in range(len(files)):
        X = pickle.load(open(files[set_no], 'rb'))
        X_test = pickle.load(open(files_test[set_no], 'rb'))
        y_true = []
        for el in Y_train:
            y_true.append(el.astype(np.dtype(np.int64)))
        y_test = []
        for el in Y_test:
            y_test.append(el.astype(np.dtype(np.int64)))
        y_tensor = torch.tensor(y_true, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        size = len(X[0])
        model = nn.Sequential(
            nn.Linear(size, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
           
        )
        optimizer = torch.optim.SGD(model.parameters())
        best_model = model
        smallest_loss = 9999999999999999999
        X_tensor = torch.Tensor(X)
        X_test_tensor = torch.Tensor(X_test)

        for n in range(num_epochs):
            y_pred = model(X_tensor)
            loss = loss_fn(y_pred, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_test_result = model(X_test_tensor)
            loss = loss_fn(y_test_result, y_test_tensor)
            print(loss.item())

            if loss.item() < smallest_loss:
                best_model = model
                smallest_loss = loss.item()

        filename = ".\models\mdl_temp_" + str(set_no)
        pickle.dump(best_model, open(filename, 'wb'))
        print("best: " + str(smallest_loss))
   
        


