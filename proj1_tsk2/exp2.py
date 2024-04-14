import numpy as np

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.inspection import DecisionBoundaryDisplay

import sklearn
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_breast_cancer, load_wine, load_iris

from sklearn.neural_network import MLPClassifier
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

def test_accuracy(X_test, y_test, model):
    correct = 0
    for k in range(len(X_test)):
        val = model.predict(X_test[k].reshape(1, -1))
        correct_option = y_test[k]

        if val == correct_option:
            correct = correct + 1

    return correct/len(X_test)