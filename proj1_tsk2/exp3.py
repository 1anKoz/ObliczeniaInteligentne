import numpy as np

import math

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

def do_mlp(x_trn, x_tst, y_trn, y_tst):

    acc_trn_arr = []
    acc_tst_arr = []
    mlp_arr = []

    my_layer_sizes = []
    i = 2
    while(i <= 60):
        #print(i)
        # file_path = ".\saved_models\MLP_" + filename + "_" + str(my_layer_sizes) + "_" + "relu" + ".sav"
        # try:
        #     loaded_model = pickle.load(open(file_path, 'rb'))
        #     return loaded_model
        # except:
        mlp = MLPClassifier(hidden_layer_sizes = i, activation="relu", max_iter=100000, tol=0, n_iter_no_change=100000, solver="sgd")
        mlp = mlp.fit(x_trn, y_trn)
        mlp_arr.append(mlp)

        y_pred_trn = mlp.predict(x_trn)
        y_pred_tst = mlp.predict(x_tst)

        acc_trn = accuracy_score(y_trn, y_pred_trn)
        acc_tst = accuracy_score(y_test, y_pred_tst)
        acc_trn_arr.append(acc_trn)
        acc_tst_arr.append(acc_tst)
        #pickle.dump(mlp_trn, open(file_path, 'wb'))
        # mlp_tst = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=my_layer_sizes, activation="relu", max_iter=100000, tol=0, n_iter_no_change=100000, solver="sgd")
        # mlp_tst = mlp_tst.fit(x_tst, y_tst)
        #pickle.dump(mlp_tst, open(file_path, 'wb'))
        my_layer_sizes.append(i)
        i += 4

    conf_mtrx_trn = confusion_matrix(y_trn, y_pred_trn)
    conf_mtrx_tst = confusion_matrix(y_tst, y_pred_tst) #use y_test instead of y_true to provide correct array size
    print("*Train: ")
    print(conf_mtrx_trn)
    print("*Test: ")
    print(conf_mtrx_tst)
    print()

    return my_layer_sizes, acc_trn_arr, acc_tst_arr, mlp_arr

def do_knn(x_trn, x_tst, y_trn, y_tst):
    n_neighbours_arr = []
    acc_trn_arr = []
    acc_tst_arr = []
    knn_arr = []

    i = 1
    while ( i <= 15 ):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(x_trn, y_trn)
        knn_arr.append(knn)

        y_pred_trn = knn.predict(x_trn)
        y_pred_tst = knn.predict(x_tst)

        acc_trn = accuracy_score(y_trn, y_pred_trn)
        acc_tst = accuracy_score(y_test, y_pred_tst)
        acc_trn_arr.append(acc_trn)
        acc_tst_arr.append(acc_tst)

        n_neighbours_arr.append(i)
        i += 1
    conf_mtrx_trn = confusion_matrix(y_trn, y_pred_trn)
    conf_mtrx_tst = confusion_matrix(y_tst, y_pred_tst) #use y_test instead of y_true to provide correct array size
    print("*Train: ")
    print(conf_mtrx_trn)
    print("*Test: ")
    print(conf_mtrx_tst)
    print()

    return n_neighbours_arr, acc_trn_arr, acc_tst_arr, knn_arr

def do_svc(x_trn, x_tst, y_trn, y_tst):
    log_c_arr = []
    acc_trn_arr = []
    acc_tst_arr = []
    svc_arr = []

    log_c = -2.0
    e = math.e
    while (log_c <= 6.0):
        c = math.pow(e, log_c)

        svc = sklearn.svm.SVC(C=c)
        svc.fit(x_trn, y_trn)   # rozważyć rozdzielenie tego na svc_trn i svc_tst,
        svc_arr.append(svc)

        y_pred_trn = svc.predict(x_trn)
        y_pred_tst = svc.predict(x_tst)

        acc_trn = accuracy_score(y_trn, y_pred_trn)
        acc_tst = accuracy_score(y_test, y_pred_tst)
        acc_trn_arr.append(acc_trn)
        acc_tst_arr.append(acc_tst)

        log_c_arr.append(log_c)
        log_c += 0.25

    conf_mtrx_trn = confusion_matrix(y_trn, y_pred_trn)
    conf_mtrx_tst = confusion_matrix(y_tst, y_pred_tst) #use y_test instead of y_true to provide correct array size
    print("*Train: ")
    print(conf_mtrx_trn)
    print("*Test: ")
    print(conf_mtrx_tst)
    print()

    return log_c_arr, acc_trn_arr, acc_tst_arr, svc_arr

def visualize_decision_boundary_2D(dataset, model, y_true, graph_title):
    feature_1, feature_2 = np.meshgrid(
        np.linspace(dataset[:, 0].min(), dataset[:, 0].max()),
        np.linspace(dataset[:, 1].min(), dataset[:, 1].max())
    )

    grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
    y_pred = np.reshape(model.predict(grid), feature_1.shape)
    display = DecisionBoundaryDisplay(
        xx0=feature_1, xx1=feature_2, response=y_pred
    )
    display.plot()
    display.ax_.scatter(
        dataset[:, 0], dataset[:, 1], c=y_true, edgecolor="black"
    )
    plt.title(graph_title)
    plt.show()



if __name__ == "__main__":

    files = ["dwbc", "iris", "wine"]#["2_2.csv", "2_3.csv"]
    for file in files:
        X, y_true = load(file)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y_true, test_size=0.2, train_size=0.2, random_state=42)
        
    # KNN
        # print("kNN confusion\nmatrix for: " + file)
        # n_neighbours_arr, knn_accuracy_training_array, knn_accuracy_test_array, model_knn_array = do_knn(X_train, X_test, y_train, y_test)

        # index_of_max_acc = np.argmax(knn_accuracy_test_array)
        # best_knn_model = model_knn_array[index_of_max_acc]
        # first_knn_model = model_knn_array[0]
        # last_knn_model = model_knn_array[len(knn_accuracy_test_array) - 1]

        # print("Max accuracy: " + str(knn_accuracy_test_array[index_of_max_acc]) + " for: " + str(file))
        # print("First accuracy: " + str(knn_accuracy_test_array[0]) + " for: " + str(file))
        # print("Last accuracy: " + str(knn_accuracy_test_array[len(knn_accuracy_test_array) - 1]) + " for: " + str(file))
        # print("---------------------------------------------------------")

        # plt.plot(n_neighbours_arr, knn_accuracy_training_array, color = 'r')
        # plt.plot(n_neighbours_arr, knn_accuracy_test_array, color = 'b')
        # plt.legend(["training accuracy", "test accuracy"])
        # plt.xlabel("n_neighbours")
        # plt.title("kNN for " + file + ". Max acc: " + str(knn_accuracy_test_array[index_of_max_acc]) + " for: " + str(n_neighbours_arr[index_of_max_acc]) + " neighbours")
        # plt.show()

        # visualize_decision_boundary_2D(X, best_knn_model, y_true, "kNN BEST decision boundary for: " + file)
        # visualize_decision_boundary_2D(X, first_knn_model, y_true, "kNN FIRST decision boundary for: " + file)
        # visualize_decision_boundary_2D(X, last_knn_model, y_true, "kNN LAST decision boundary for: " + file)

        
    # SVC
        # print("SVC confusion\nmatrix for: " + file)
        # log_c_array, svc_accuracy_training_array, svc_accuracy_test_array, model_svc_array = do_svc(X_train, X_test, y_train, y_test)

        # index_of_max_acc = np.argmax(svc_accuracy_test_array)
        # best_svc_model = model_svc_array[index_of_max_acc]
        # first_svc_model = model_svc_array[0]
        # last_svc_model = model_svc_array[len(svc_accuracy_test_array) - 1]

        # print("Max accuracy: " + str(svc_accuracy_test_array[index_of_max_acc]) + " for: " + str(file))
        # print("First accuracy: " + str(svc_accuracy_test_array[0]) + " for: " + str(file))
        # print("Last accuracy: " + str(svc_accuracy_test_array[len(svc_accuracy_test_array) - 1]) + " for: " + str(file))
        # print("---------------------------------------------------------")

        # plt.plot(log_c_array, svc_accuracy_training_array, color = 'r')
        # plt.plot(log_c_array, svc_accuracy_test_array, color = 'b')
        # plt.legend(["training accuracy", "test accuracy"])
        # plt.xlabel("log(c)")
        # plt.title("SVC for " + file + ". Max acc: " + str(svc_accuracy_test_array[index_of_max_acc]) + " for 'C': " + str(math.pow(math.e, log_c_array[index_of_max_acc])))
        # plt.show()

        # visualize_decision_boundary_2D(X, best_svc_model, y_true, "SVC BEST decision boundary for: " + file)
        # visualize_decision_boundary_2D(X, first_svc_model, y_true, "SVC FIRST decision boundary for: " + file)
        # visualize_decision_boundary_2D(X, last_svc_model, y_true, "SVC LAST decision boundary for: " + file)


    #MLP
        print("MLP confusion\nmatrix for: " + file)
        n_of_neurons_array, mlp_accuracy_train_array, mlp_accuracy_test_array, model_mlp_array = do_mlp(X_train, X_test, y_train, y_test)

        index_of_max_acc = np.argmax(mlp_accuracy_test_array)
        best_mlp_model = model_mlp_array[index_of_max_acc]
        first_mlp_model = model_mlp_array[0]
        last_mlp_model = model_mlp_array[len(mlp_accuracy_test_array) - 1]

        print("Max accuracy: " + str(mlp_accuracy_test_array[index_of_max_acc]) + " for: " + str(file))
        print("First accuracy: " + str(mlp_accuracy_test_array[0]) + " for: " + str(file))
        print("Last accuracy: " + str(mlp_accuracy_test_array[len(mlp_accuracy_test_array) - 1]) + " for: " + str(file))
        print("---------------------------------------------------------")

        # plt.plot(n_of_neurons_array, mlp_accuracy_train_array, color = 'r')
        # plt.plot(n_of_neurons_array, mlp_accuracy_test_array, color = 'b')
        # plt.legend(["training accuracy", "test accuracy"])
        # plt.xlabel("n of neurons")
        # plt.title("MLP for " + file + ". Max acc: " + str(mlp_accuracy_test_array[index_of_max_acc]) + " for: " + str(n_of_neurons_array[index_of_max_acc]) + " neurons")
        # plt.show()

        # visualize_decision_boundary_2D(X, best_mlp_model, y_true, "MLP BEST decision boundary for: " + file)
        # visualize_decision_boundary_2D(X, first_mlp_model, y_true, "MLP FIRST decision boundary for: " + file)
        # visualize_decision_boundary_2D(X, last_mlp_model, y_true, "MLP LAST decision boundary for: " + file)