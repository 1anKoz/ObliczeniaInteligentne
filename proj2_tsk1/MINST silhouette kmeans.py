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

from keras.datasets import mnist
import pickle


def voronoi_finite_polygons_2d(vor, radius=None):
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp(axis=0).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot_voronoi_diagram(X, y_true, y_pred, title):
    colors = []
    for c in matplotlib.colors.get_named_colors_mapping():
        colors.append(c)
    j = 0

    # compute Voronoi tesselation
    vor = Voronoi(X)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    # colorize
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4, color=colors[y_pred[j] + 581])
        j += 1

    for a in vor.points:
        index = int(np.where(X==a)[0][0])
        plt.scatter(a[0],a[1], color=colors[int(y_true[index ] + 4)],s=4)

    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

    plt.title(title)
    plt.show()

def clustering_KMeans(points, n_of_clusters):
    algorithm = cluster.KMeans(n_clusters = n_of_clusters, n_init = 'auto')
    algorithm.fit(points)
    fake_labels_arr = algorithm.labels_.astype(int)
    return fake_labels_arr

def silhouette(points_arr, max_n_of_clusters):
    i = 2
    score = np.array([])
    while i < max_n_of_clusters:
        fake_labels_arr = clustering_KMeans(points_arr, i)
        sil_score = silhouette_score(points_arr, fake_labels_arr)
        score = np.append(score, sil_score)
        i += 1
    return score

if __name__ == "__main__":

    # reading the data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_2 = pickle.load(open(".\encodings\J_enc_train_2_features.sav", 'rb'))
    # /|\ zmieniać
    points_arr1 = X_2
    true_labels_arr1 = Y_train

    # preparing matrix of numbers of clusters to properly plot a silhouette
    n_of_clusters = 10
    x = 2
    n_of_clusters_arr = []
    while x < n_of_clusters:
        n_of_clusters_arr = np.append(n_of_clusters_arr, x)
        x += 1

    sil_score1 = silhouette(points_arr1, n_of_clusters)
    
    # silhouette plots
    plt.plot(n_of_clusters_arr, sil_score1, color = 'green', marker = 'o')
    plt.xlabel ('n_clusters')
    plt.show()
   

    # best and worst silhouette cases
    best_case1 = np.argmax(sil_score1) + 2
    

    worst_case1 = np.argmin(sil_score1) + 2
   
    # clustering and voronoi best case
    fake_labels_arr1 = clustering_KMeans(points_arr1, best_case1)
   

    title1 = "Train_2 KMeans best case " + str(best_case1) + " clusters"
    plot_voronoi_diagram(points_arr1, true_labels_arr1, fake_labels_arr1, title1)
    
    
