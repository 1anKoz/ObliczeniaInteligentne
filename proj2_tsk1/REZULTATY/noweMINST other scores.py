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

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
import pickle
from sklearn import preprocessing

import os


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

    # compute Voronoi tesselation
    vor = Voronoi(X)
    j = 0

    # plot
    if y_true is not None:
        voronoi_plot_2d(vor, show_points=False, show_vertices=False)
    else:
        voronoi_plot_2d(vor, show_points=True, show_vertices=False)

    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            if y_pred[j] < len(colors):
                plt.fill(*zip(*polygon),color=colors[y_pred[j]])
            else:
                plt.fill(*zip(*polygon),color="black")
        j += 1

    if y_true is not None:
        for a in vor.points:
            index = int(np.where(X==a)[0][0])
            plt.scatter(a[0],a[1],color=colors[int(y_true[index])],s=4)

    plt.title(title)
    plt.savefig("./DBSCAN_v.png")
    #plt.show()


if __name__ == "__main__":
    #files = [".\encodings\enc_train_784.sav", ".\encodings\enc_train_our.sav", ".\encodings\enc_train_2.sav"]
    #files = [".\encodings\J_enc_train_2_features.sav"]
    files = [".\encodings\enc_train_2.sav"]
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    for set_no in range(len(files)):
        X = pickle.load(open(files[set_no], 'rb'))
        #X = preprocessing.normalize(X)
        y_true = Y_train
   
    
    #DBSCAN
        temp_esp_list = np.linspace(1, 30, 15, endpoint=False).tolist()
        esp_options = [ round(elem, 2) for elem in temp_esp_list ]
        adjusted_rand_points = []
        homogeneity_points = []
        completeness_points = []
        V_measure5_points = []
        V_measure1_points = []
        V_measure2_points = []
        goodness_score_table = []
        y_pred_table = []
        no_clusters_table = []
        i = 0
        for option in esp_options:
            algorithm = DBSCAN(eps=option, min_samples=1)
            algorithm.fit(X)
            y_pred = algorithm.labels_.astype(int)
            adjusted_rand_points.append(sklearn.metrics.adjusted_rand_score(y_true, y_pred))
            homogeneity_points.append(sklearn.metrics.homogeneity_score(y_true, y_pred))
            completeness_points.append(sklearn.metrics.completeness_score(y_true, y_pred))
            V_measure5_points.append(sklearn.metrics.v_measure_score(y_true, y_pred, beta=0.5))
            V_measure2_points.append(sklearn.metrics.v_measure_score(y_true, y_pred, beta=2))
            V_measure1_points.append(sklearn.metrics.v_measure_score(y_true, y_pred, beta=1))
            goodness_score = adjusted_rand_points[i] + V_measure1_points[i]
            goodness_score_table.append(goodness_score)
            y_pred_table.append(y_pred)
            no_clusters = len(set(algorithm.labels_)) - (1 if -1 in algorithm.labels_ else 0)
            no_clusters_table.append(no_clusters)
            plt.plot([option, option],[0,1],color='grey', linestyle='--', linewidth=1)
            plt.text(option, 0.075, str(no_clusters))
            i += 1

        plt.plot(esp_options, adjusted_rand_points, 'blue', label='adjusted rand')
        plt.plot(esp_options, homogeneity_points, 'orange', label='homogenity')
        plt.plot(esp_options, completeness_points, 'green', label='completeness')
        plt.plot(esp_options, V_measure5_points, 'pink', label='V-measure (beta=0,5)')
        plt.plot(esp_options, V_measure2_points, 'purple', label='V-measure (beta=2)')
        plt.plot(esp_options, V_measure1_points, 'brown', label='V-measure (beta=1)')
        plt.legend(loc="upper right")
        plt.xlabel ('eps')
        plt.title(files[set_no] + " DBSCAN")
        filen = "./plot_DBSCAN_" + str(set_no) + ".png"
        #plt.savefig(filen)
        plt.show()
        

        if "_2" in files[set_no]:
            min_goodness = min(goodness_score_table)
            max_goodness = max(goodness_score_table)
            shown_good = False
            shown_bad = False
            
            index = goodness_score_table.index(max_goodness)
            print("For DBSCAN of " + files[set_no] + " best case was found to be eps = " + str(esp_options[index]) + " number of clusters: " + str(no_clusters_table[index]))
            
            for j in range(len(goodness_score_table)):
                if (not shown_good) and goodness_score_table[j] == max_goodness:
                    plot_voronoi_diagram(X, y_true, y_pred_table[j], "Best case for DBSCAN of " + files[set_no] + " eps = " + str(esp_options[j]) + " number of clusters: " + str(no_clusters_table[j]))
                    shown_good = True
        else:
            max_goodness = max(goodness_score_table)
            index = goodness_score_table.index(max_goodness)
            print("For DBSCAN of " + files[set_no] + " best case was found to be eps = " + str(esp_options[index]) + " number of clusters: " + str(no_clusters_table[index]))

        continue
    #K-means
        no_clusters = [2,3,4,5,6,7,8,9,10,11]
        adjusted_rand_points = []
        homogeneity_points = []
        completeness_points = []
        V_measure5_points = []
        V_measure1_points = []
        V_measure2_points = []
        goodness_score_table = []
        y_pred_table = []
        i = 0
        for option in no_clusters:
            algorithm = cluster.KMeans(n_clusters=option)
            algorithm.fit(X)
            y_pred = algorithm.labels_.astype(int)
            adjusted_rand_points.append(sklearn.metrics.adjusted_rand_score(y_true, y_pred))
            homogeneity_points.append(sklearn.metrics.homogeneity_score(y_true, y_pred))
            completeness_points.append(sklearn.metrics.completeness_score(y_true, y_pred))
            V_measure5_points.append(sklearn.metrics.v_measure_score(y_true, y_pred, beta=0.5))
            V_measure2_points.append(sklearn.metrics.v_measure_score(y_true, y_pred, beta=2))
            V_measure1_points.append(sklearn.metrics.v_measure_score(y_true, y_pred, beta=1))
            goodness_score = adjusted_rand_points[i] + V_measure1_points[i]
            goodness_score_table.append(goodness_score)
            y_pred_table.append(y_pred)
            i += 1
            
        plt.plot(no_clusters, adjusted_rand_points, 'blue', label='adjusted rand')
        plt.plot(no_clusters, homogeneity_points, 'orange', label='homogenity')
        plt.plot(no_clusters, completeness_points, 'green', label='completeness')
        plt.plot(no_clusters, V_measure5_points, 'pink', label='V-measure (beta=0,5)')
        plt.plot(no_clusters, V_measure2_points, 'purple', label='V-measure (beta=2)')
        plt.plot(no_clusters, V_measure1_points, 'brown', label='V-measure (beta=1)')
        plt.legend(loc="upper right")
        plt.xlabel ('n_clusters')
        plt.grid(axis='x', which='major', color='grey', linestyle='--', linewidth=1)
        plt.title(files[set_no] + " K Means")
        plt.show()

        if "x" in files[set_no]:
            min_goodness = min(goodness_score_table)
            max_goodness = max(goodness_score_table)
            shown_good = False
            shown_bad = False
            for j in range(len(goodness_score_table)):
                if (not shown_good) and goodness_score_table[j] == max_goodness:
                    plot_voronoi_diagram(X, y_true, y_pred_table[j], "Best case for K Means of " + files[set_no] + " number of clusters: " + str(no_clusters[j]))
                    shown_good = True
        else:
            max_goodness = max(goodness_score_table)
            index = goodness_score_table.index(max_goodness)
            print("For K Means of " + files[set_no] + " best case was found to be " + str(no_clusters[index]) + " clusters ")

        
