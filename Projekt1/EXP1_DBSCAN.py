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


def csv_read(path):
    points = pd.read_csv(path, sep = ";", usecols=[0,1])
    labels = pd.read_csv(path, sep = ";", usecols=[2])

    points_arr = np.array(points)
    labels_arr = np.array(labels)

    return points_arr, labels_arr

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
    points_arr1, true_labels_arr1 = csv_read("1_1.csv")
    points_arr2, true_labels_arr2 = csv_read("1_2.csv")
    points_arr3, true_labels_arr3 = csv_read("1_3.csv")
    points_arr4, true_labels_arr4 = csv_read("2_1.csv")
    points_arr5, true_labels_arr5 = csv_read("2_2.csv")
    points_arr6, true_labels_arr6 = csv_read("2_3.csv")

    # preparing the matrix of epsilons to properly plot a silhouette
    epsilon = 2.00
    x = 0.10
    epsilon_arr = []
    while x <= epsilon:
        epsilon_arr = np.append(epsilon_arr, x)
        x += 0.10
    sil_score1, n_of_DBSCAN_clusters1 = silhouette(points_arr1, epsilon)
    sil_score2, n_of_DBSCAN_clusters2 = silhouette(points_arr2, epsilon)
    sil_score3, n_of_DBSCAN_clusters3 = silhouette(points_arr3, epsilon)
    sil_score4, n_of_DBSCAN_clusters4 = silhouette(points_arr4, epsilon)
    sil_score5, n_of_DBSCAN_clusters5 = silhouette(points_arr5, epsilon)
    sil_score6, n_of_DBSCAN_clusters6 = silhouette(points_arr6, epsilon)

    # print('********************')
    # print(sil_score3)
    # print(np.argmax(sil_score3))
    # print(n_of_DBSCAN_clusters3)

    # silhouette plots
    fig, ax = plt.subplots()
    ax.scatter(epsilon_arr, sil_score1)
    for i, txt in enumerate(n_of_DBSCAN_clusters1):
        ax.annotate(txt, (epsilon_arr[i], sil_score1[i]))
    plt.plot(epsilon_arr, sil_score1, color = 'green', marker = 'o')
    plt.xlabel ('eps')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(epsilon_arr, sil_score2)
    for i, txt in enumerate(n_of_DBSCAN_clusters2):
        ax.annotate(txt, (epsilon_arr[i], sil_score2[i]))
    plt.plot(epsilon_arr, sil_score2, color = 'green', marker = 'o')
    plt.xlabel ('eps')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(epsilon_arr, sil_score3)
    for i, txt in enumerate(n_of_DBSCAN_clusters3):
        ax.annotate(txt, (epsilon_arr[i], sil_score3[i]))
    plt.plot(epsilon_arr, sil_score3, color = 'green', marker = 'o')
    plt.xlabel ('eps')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(epsilon_arr, sil_score4)
    for i, txt in enumerate(n_of_DBSCAN_clusters4):
        ax.annotate(txt, (epsilon_arr[i], sil_score4[i]))
    plt.plot(epsilon_arr, sil_score4, color = 'green', marker = 'o')
    plt.xlabel ('eps')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(epsilon_arr, sil_score5)
    for i, txt in enumerate(n_of_DBSCAN_clusters5):
        ax.annotate(txt, (epsilon_arr[i], sil_score5[i]))
    plt.plot(epsilon_arr, sil_score5, color = 'green', marker = 'o')
    plt.xlabel ('eps')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(epsilon_arr, sil_score6)
    for i, txt in enumerate(n_of_DBSCAN_clusters6):
        ax.annotate(txt, (epsilon_arr[i], sil_score6[i]))
    plt.plot(epsilon_arr, sil_score6, color = 'green', marker = 'o')
    plt.xlabel ('eps')
    plt.show()

    # best and worst silhouette cases
    best_case1 = epsilon_arr[np.argmax(sil_score1)]
    best_case2 = epsilon_arr[np.argmax(sil_score2)]
    best_case3 = epsilon_arr[np.argmax(sil_score3)]
    best_case4 = epsilon_arr[np.argmax(sil_score4)]
    best_case5 = epsilon_arr[np.argmax(sil_score5)]
    best_case6 = epsilon_arr[np.argmax(sil_score6)]

    worst_case1 = epsilon_arr[np.argmin(sil_score1)]
    worst_case2 = epsilon_arr[np.argmin(sil_score2)]
    worst_case3 = epsilon_arr[np.argmin(sil_score3)]
    worst_case4 = epsilon_arr[np.argmin(sil_score4)]
    worst_case5 = epsilon_arr[np.argmin(sil_score5)]
    worst_case6 = epsilon_arr[np.argmin(sil_score6)]

    # clustering and voronoi best case
    fake_labels_arr1, n_of_DBSCAN_clusters1 = clustering_DBscan(points_arr1, best_case1)
    fake_labels_arr2, n_of_DBSCAN_clusters2 = clustering_DBscan(points_arr2, best_case2)
    fake_labels_arr3, n_of_DBSCAN_clusters3 = clustering_DBscan(points_arr3, best_case3)
    fake_labels_arr4, n_of_DBSCAN_clusters4 = clustering_DBscan(points_arr4, best_case4)
    fake_labels_arr5, n_of_DBSCAN_clusters5 = clustering_DBscan(points_arr5, best_case5)
    fake_labels_arr6, n_of_DBSCAN_clusters6 = clustering_DBscan(points_arr6, best_case6)



    title1 = "1_1.csv DBscan best case, epsilon: " + str(best_case1) + " clusters: " + str(n_of_DBSCAN_clusters1)
    plot_voronoi_diagram(points_arr1, true_labels_arr1, fake_labels_arr1, title1)
    title2 = "1_2.csv DBscan best case, epsilon: " + str(best_case2) + " clusters: " + str(n_of_DBSCAN_clusters2)
    plot_voronoi_diagram(points_arr2, true_labels_arr2, fake_labels_arr2, title2)
    title3 = "1_3.csv DBscan best case, epsilon: " + str(best_case3) + " clusters: " + str(n_of_DBSCAN_clusters3)
    plot_voronoi_diagram(points_arr3, true_labels_arr3, fake_labels_arr3, title3)
    title4 = "2_1.csv DBscan best case, epsilon: " + str(best_case4) + " clusters: " + str(n_of_DBSCAN_clusters4)
    plot_voronoi_diagram(points_arr4, true_labels_arr4, fake_labels_arr4, title4)
    title5 = "2_2.csv DBscan best case, epsilon: " + str(best_case5) + " clusters: " + str(n_of_DBSCAN_clusters5)
    plot_voronoi_diagram(points_arr5, true_labels_arr5, fake_labels_arr5, title5)
    title6 = "2_3.csv DBscan best case, epsilon: " + str(best_case6) + " clusters: " + str(n_of_DBSCAN_clusters6)
    plot_voronoi_diagram(points_arr6, true_labels_arr6, fake_labels_arr6, title6)

    #clustering and voronoi worst case
    fake_labels_arr1, n_of_DBSCAN_clusters1 = clustering_DBscan(points_arr1, worst_case1)
    fake_labels_arr2, n_of_DBSCAN_clusters2 = clustering_DBscan(points_arr2, worst_case2)
    fake_labels_arr3, n_of_DBSCAN_clusters3 = clustering_DBscan(points_arr3, worst_case3)
    fake_labels_arr4, n_of_DBSCAN_clusters4 = clustering_DBscan(points_arr4, worst_case4)
    fake_labels_arr5, n_of_DBSCAN_clusters5 = clustering_DBscan(points_arr5, worst_case5)
    fake_labels_arr6, n_of_DBSCAN_clusters6 = clustering_DBscan(points_arr6, worst_case6)

    title1 = "1_1.csv DBscan worst case, epsilon:  " + str(worst_case1) + " clusters: " + str(n_of_DBSCAN_clusters1)
    plot_voronoi_diagram(points_arr1, true_labels_arr1, fake_labels_arr1, title1)
    title2 = "1_2.csv DBscan worst case, epsilon:  " + str(worst_case2) + " clusters: " + str(n_of_DBSCAN_clusters2)
    plot_voronoi_diagram(points_arr2, true_labels_arr2, fake_labels_arr2, title2)
    title3 = "1_3.csv DBscan worst case, epsilon:  " + str(worst_case3) + " clusters: " + str(n_of_DBSCAN_clusters3)
    plot_voronoi_diagram(points_arr3, true_labels_arr3, fake_labels_arr3, title3)
    title4 = "2_1.csv DBscan worst case, epsilon:  " + str(worst_case4) + " clusters: " + str(n_of_DBSCAN_clusters4)
    plot_voronoi_diagram(points_arr4, true_labels_arr4, fake_labels_arr4, title4)
    title5 = "2_2.csv DBscan worst case, epsilon:  " + str(worst_case5) + " clusters: " + str(n_of_DBSCAN_clusters5)
    plot_voronoi_diagram(points_arr5, true_labels_arr5, fake_labels_arr5, title5)
    title6 = "2_3.csv DBscan worst case, epsilon:  " + str(worst_case6) + " clusters: " + str(n_of_DBSCAN_clusters6)
    plot_voronoi_diagram(points_arr6, true_labels_arr6, fake_labels_arr6, title6)