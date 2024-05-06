import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import pickle
import numpy as np

if __name__ == "__main__":
    files = [".\encodings\enc_train_784.sav", ".\encodings\enc_train_our.sav", ".\encodings\enc_train_2.sav"]
    #files = [".\encodings\enc_train_our.sav"]
    #files_test = [".\encodings\enc_test_our.sav"]
    files_test = [".\encodings\enc_test_784.sav", ".\encodings\enc_test_our.sav", ".\encodings\enc_test_2.sav"]
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
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
            #print(loss.item())

            if loss.item() < smallest_loss:
                best_model = model
                smallest_loss = loss.item()

        filename = ".\models\mdl_temp_" + str(set_no)
        pickle.dump(best_model, open(filename, 'wb'))
        print("best: " + str(smallest_loss))
   
        


