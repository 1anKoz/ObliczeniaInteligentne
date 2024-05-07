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

def try_to_fill(x, y, mask):
    if x < 27 and mask[x+1][y] == 0:
        mask[x+1][y] = 123
        try_to_fill(x+1, y, mask)

    if x > 0 and mask[x-1][y] == 0:
        mask[x-1][y] = 123
        try_to_fill(x-1, y, mask)

    if y < 27 and mask[x][y+1] == 0:
        mask[x][y+1] = 123
        try_to_fill(x, y+1, mask)

    if y > 0 and mask[x][y-1] == 0:
        mask[x][y-1] = 123
        try_to_fill(x, y-1, mask)

def encoding_2(data):
    vector = []
    masks = []
    for row in data:
        no_edge = 0
        y_current = 0
        y_new = 1
        value_final = 0
        mask = row.copy()
        try_to_fill(0, 0, mask)
        for x in range(1,len(row)-1):
            for y in range(1, len(row[0])-1):
                if row[x][y-1] == 0 or row[x][y+1] == 0 or row[x+1][y] == 0 or row[x-1][y] == 0:
                    no_edge = no_edge + 1

                if mask[x][y] == 0:
                    if not y == y_current:
                        if y == y_new:
                            y_current = y
                            y_new = y_new + 1
                        else:
                            value_final = value_final + y_current
                            y_current = y
                            y_new = y + 1
                        
        vector.append([no_edge, value_final])
        masks.append(mask)
    return vector, masks

def width_measurements(data):
    vectors = []
    for row in data:
        measureBlack = False
        separated = False
        val = 0
        if not row[0] == 0:
            measureBlack = True

        for el in row:
            if not el == 0:
                if measureBlack:
                    val += 1
                else:
                    if val > 0:
                        separated = True
                    measureBlack = True
                    val += 1
            else:
                measureBlack = False

        if separated:
            val *= -1
        vectors.append(val)
    return vectors
    
def encoding_our(data):
    vectors = []
    for el in data:
        vectors.append(width_measurements(el))
    return vectors
    

def get_MINST_encoded_1(data):
    vector_784 = data.reshape(len(data), 784)
    vector_2, masks = encoding_2(data)
    vector_our = encoding_our(data)
    return vector_784, vector_2, vector_our

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    vector_784, vector_2, vector_our = get_MINST_encoded_1(X_test)
    print("**********************")
    print(vector_2)
    print("-----------------------------")
    print(vector_our)
    file_path = ".\encodings\enc_test_784.sav"
    pickle.dump(vector_784, open(file_path, 'wb'))
    file_path = ".\encodings\enc_test_2.sav"
    pickle.dump(vector_2, open(file_path, 'wb'))
    file_path = ".\encodings\enc_test_our.sav"
    pickle.dump(vector_our, open(file_path, 'wb'))
    
    vector_784, vector_2, vector_our = get_MINST_encoded_1(X_train)
    file_path = ".\encodings\enc_train_784.sav"
    pickle.dump(vector_784, open(file_path, 'wb'))
    file_path = ".\encodings\enc_train_2.sav"
    pickle.dump(vector_2, open(file_path, 'wb'))
    file_path = ".\encodings\enc_train_our.sav"
    pickle.dump(vector_our, open(file_path, 'wb'))

        



    



        
