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

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import math

def explain(encoding, num, values):
    print(str(num) + " is encoded as:")
    print(encoding)
    plt.imshow(values.reshape(28, 28), cmap = "binary")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    files = [".\encodings\J_enc_train_2_features.sav", ".\encodings\J_enc_train_8_features.sav"]
    for set_no in range(len(files)):
        encodings = pickle.load(open(files[set_no], 'rb'))
        show_8 = False
        show_0 = False
        show_1 = False
        show_3 = False
        i = 0
        for el in encodings:
            if Y_test[i] == 8 and not show_8:
                explain(el, Y_test[i], X_test[i])
                show_8 = True

            if Y_test[i] == 0 and not show_0:
                explain(el, Y_test[i], X_test[i])
                show_0 = True

            if Y_test[i] == 1 and (not show_1):
                explain(el, Y_test[i], X_test[i])
                show_1 = True

            if Y_test[i] == 3 and not show_3:
                explain(el, Y_test[i], X_test[i])
                show_3 = True
                                
            i+=1

        



    



        
