import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import pickle
import numpy as np
from iris_load import IrisDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

if __name__ == "__main__":
    #files = [".\encodings\enc_train_2.sav", ".\encodings\enc_train_784.sav", ".\encodings\enc_train_our.sav"]
    files = [".\encodings\iris_train.sav"]
    X_train, X_test, y_train, y_test = train_test_split(load_iris().data, load_iris().target, test_size=0.2, random_state=42)
    #(X_train, y_train), (X_test, Y_test) = mnist.load_data()
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 300
    for set_no in range(len(files)):
        X = pickle.load(open(files[set_no], 'rb'))
        y_true = []
        for el in y_train:
            y_true.append(el.astype(np.dtype(np.int64)))
        y_tensor = torch.tensor(y_true, dtype=torch.long)
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

        for n in range(num_epochs):
            y_pred = model(X_tensor)
            loss = loss_fn(y_pred, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())

            if loss.item() < smallest_loss:
                best_model = model
                smallest_loss = loss.item()

        pickle.dump(best_model, open(".\models\iris_train", 'wb'))
        print("best: " + str(smallest_loss))
   
        


