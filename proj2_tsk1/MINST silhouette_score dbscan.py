import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn import cluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_breast_cancer, load_wine, load_iris

from itertools import groupby

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
import pickle

def clustering_KMeans(points, n_of_clusters):
    algorithm = cluster.KMeans(n_clusters = n_of_clusters, n_init = 'auto')
    algorithm.fit(points)
    fake_labels_arr = algorithm.labels_.astype(int)
    return fake_labels_arr

def clustering_DBscan(points, epsilon):
    algorithm = cluster.DBSCAN(eps = epsilon, min_samples = 1)
    algorithm.fit(points)
    fake_labels_arr = algorithm.labels_.astype(int)
    n_of_DBSCAN_clusters = len(np.unique(fake_labels_arr))
    return fake_labels_arr, n_of_DBSCAN_clusters

def silhouette(points_arr, max_n):
    i = 0.10
    score = np.array([])
    DBSCAN_clusters_array = []
    while i <= max_n:
        fake_labels_arr, n_of_DBSCAN_clusters = clustering_DBscan(points_arr, i)

        if all_equal(fake_labels_arr):
            score = np.append(score, 0)
            i += 0.10
            continue

        sil_score = silhouette_score(points_arr, fake_labels_arr)
        score = np.append(score, sil_score)

        DBSCAN_clusters_array = np.append(DBSCAN_clusters_array, n_of_DBSCAN_clusters)
        DBSCAN_clusters_array = DBSCAN_clusters_array.astype(int)
        i += 0.10
    return score, DBSCAN_clusters_array

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)



if __name__ == "__main__":

    # reading the data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_test_2 = pickle.load(open(".\encodings\enc_train_2.sav", 'rb'))
    # /|\ zmieniać


    # preparing the matrix of epsilons to properly plot a silhouette
    epsilon = 2.00
    x = 0.10
    epsilon_arr = []
    while x <= epsilon:
        epsilon_arr = np.append(epsilon_arr, x)
        x += 0.10
    sil_score1, n_of_DBSCAN_clusters1 = silhouette(X_test_2, epsilon)


    fig, ax = plt.subplots()
    ax.scatter(epsilon_arr, sil_score1)
    for i, txt in enumerate(n_of_DBSCAN_clusters1):
        ax.annotate(txt, (epsilon_arr[i], sil_score1[i]))
    plt.plot(epsilon_arr, sil_score1, color = 'green', marker = 'o')
    plt.title("Shilluete score - DBSCAN - 2")
    plt.xlabel ('eps')
    plt.show()
