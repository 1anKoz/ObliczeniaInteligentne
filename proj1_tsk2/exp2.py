import numpy as np

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Voronoi, voronoi_plot_2d

import pickle
import os
import time

def loadDWBC():
    data = load_breast_cancer()
    ys = data.target
    exes = data.data
    return exes, np.ravel(ys)

def loadIris():
    data = load_iris()
    ys = data.target
    exes = data.data
    return exes, np.ravel(ys)

def loadWine():
    data = load_wine()
    ys = data.target
    exes = data.data
    return exes, np.ravel(ys)

def load(path):
    if path == "dwbc":
        return loadDWBC()
    if path == "iris":
        return loadIris()
    if path == "wine":
        return loadWine()
    full_array = np.loadtxt(path, delimiter = ';')
    size = full_array.shape[1]
    exes = full_array[:,:(size - 1)]
    ys = full_array[:,(size - 1):]
    return exes, np.ravel(ys)

def do_mlp(my_layer_sizes, my_activation, X_train, y_train, filename):
    file_path = ".\saved_models\MLP_" + filename + "_" + str(my_layer_sizes) + "_" + my_activation + ".sav"
    try:
        loaded_model = pickle.load(open(file_path, 'rb'))
        return loaded_model
    except:
        model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=my_layer_sizes, activation=my_activation, max_iter=100000, tol=0, n_iter_no_change=100000, solver="sgd")
        model = model.fit(X_train, y_train)
        pickle.dump(model, open(file_path, 'wb'))
        return model
    
def do_svc(my_kernel, my_c, X_train, y_train):
    model = sklearn.svm.SVC(C=my_c, kernel=my_kernel)
    model = model.fit(X_train, y_train)
    return model

def do_knn(x_trn, x_tst, y_trn, y_tst):
    n_neighbours_arr = []
    acc_trn_arr = []
    acc_tst_arr = []
    i = 1
    while ( i <= 15 ):
        n_neighbours_arr.append(i)
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(x_trn, y_trn)
        y_pred_trn = knn.predict(x_trn)
        y_pred_tst = knn.predict(x_tst)
        acc_trn = accuracy_score(y_trn, y_pred_trn)
        acc_tst = accuracy_score(y_test, y_pred_tst)
        acc_trn_arr.append(acc_trn)
        acc_tst_arr.append(acc_tst)
        i += 1
    conf_mtrx_trn = confusion_matrix(y_trn, y_pred_trn)
    conf_mtrx_tst = confusion_matrix(y_tst, y_pred_tst) #use y_test instead of y_true to provide correct array size
    print("Confusion matrix train: ")
    print(conf_mtrx_trn)
    print("Confusion matrix test: ")
    print(conf_mtrx_tst)
    return n_neighbours_arr, acc_trn_arr, acc_tst_arr#, conf_mtrx_trn, conf_mtrx_tst


# def test_accuracy(X_test, y_test, model):
#     correct = 0
#     for k in range(len(X_test)):
#         val = model.predict(X_test[k].reshape(1, -1))
#         correct_option = y_test[k]

#         if val == correct_option:
#             correct = correct + 1

#     return correct/len(X_test)



if __name__ == "__main__":

    files = ["2_2.csv", "2_3.csv"]
    for file in files:
        X, y_true = load(file)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y_true, test_size=0.2, train_size=0.8, random_state=42)
        # KNN
        n_neighbours_arr, accuracy_training_array, accuracy_test_array = do_knn(X_train, X_test, y_train, y_test)

        plt.plot(n_neighbours_arr, accuracy_training_array, color = 'r')
        plt.plot(n_neighbours_arr, accuracy_test_array, color = 'b')
        plt.legend(["training accuracy", "test accuracy"])
        plt.xlabel("n_neighbours")
        plt.title("kNN for " + file)
        plt.show()
        # print("++++++++++++++++" + file + "++++++++++++++++++")
        # print(n_neighbours_arr)
        # print("************")
        # print(accuracies_arr)
        # print("---------------------------")