import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import pickle
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import sklearn
import matplotlib.pyplot as plt

def visualize_decision_boundary_2D(dataset, model, y_true, graph_title):
    min_0 = 9999999999999999999
    min_1 = 0000000000000000000
    max_0 = 0
    max_1 = 0

    dataset_0 = []
    dataset_1 = []
    for el in dataset:
        if el[0] > max_0:
            max_0 = el[0]
        if el[0] < min_0:
            min_0 = el[0]
        dataset_0.append(el[0])

        if el[1] > max_1:
            max_1 = el[1]
        if el[1] < min_1:
            min_1 = el[1]
        dataset_1.append(el[1])
            
    feature_1, feature_2 = np.meshgrid(
        np.linspace(min_0, max_0),
        np.linspace(min_1, max_1)
    )

    grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
    
    X_test_tensor = torch.Tensor(grid)
    with torch.no_grad():
        y_pred = model(X_test_tensor)
    _, y_pred = torch.max(y_pred, 1)
        
    y_pred = np.reshape(y_pred, feature_1.shape)
    display = DecisionBoundaryDisplay(
        xx0=feature_1, xx1=feature_2, response=y_pred
    )
    display.plot()
    display.ax_.scatter(
        dataset_0, dataset_1, c=y_true, edgecolor="black"
    )
    plt.title(graph_title)
    plt.show()

if __name__ == "__main__":
    files = [".\models\mdl_train_784.sav", ".\models\mdl_train_our.sav", ".\models\mdl_train_2.sav"]
    files_test = [".\encodings\enc_test_784.sav", ".\encodings\enc_test_our.sav", ".\encodings\enc_test_2.sav"]
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    for set_no in range(len(files)):
        model = pickle.load(open(files[set_no], 'rb'))
        model.eval()
        result_np = []
        X_test = pickle.load(open(files_test[set_no], 'rb'))
        y_test = []
        for el in Y_test:
            y_test.append(el.astype(np.dtype(np.int64)))
        y_tensor = torch.tensor(y_test, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        X_test_tensor = torch.Tensor(X_test)
        with torch.no_grad():
            y_pred = model(X_test_tensor)
        _, predicted_class = torch.max(y_pred, 1)

        conf_mtrx_tst = confusion_matrix(y_test, predicted_class)
        print("*Macierz pomyłek dla " + files[set_no])
        print(conf_mtrx_tst)

        acc_tst = accuracy_score(y_test, predicted_class)
        print("Dokładność:")
        print(acc_tst)

        if "_2" in files[set_no]:
            visualize_decision_boundary_2D(X_test, model, y_test, "Decision boundry for two element encoding")

        
   
        


