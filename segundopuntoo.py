import numpy as np
import matplotlib . pyplot as plt
import time
from sklearn import cluster
from scipy.io import arff
from sklearn import metrics
from sklearn.metrics import pairwise_distances, silhouette_samples
import matplotlib.cm as cm
import time
#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension

# f1 : valeur sur la deuxieme dimension
#

path = './artificial/'
 
## Load differents datasets

## ---------------Datasets with evident clusters------------------------

databrut = arff.loadarff(open(path+"2d-4c.arff" , 'r') )
#databrut = arff.loadarff (open(path+"xclara.arff" , 'r') )
#databrut = arff.loadarff (open(path+"cure-t2-4k.arff" , 'r') )

## ______________________________________________________________________


## --------------Dataset without evident classes-----------------

#databrut = arff.loadarff (open(path+"spiral.arff" , 'r') )
#databrut = arff.loadarff (open(path+"3-spiral.arff" , 'r') )

datanp = [ [ x[0] ,x[1]] for x in databrut [ 0 ] ]

f0 = [x[0] for x in datanp]
f1 = [x[1] for x in datanp]

##-------------- bucle for change init k ------------
range_n_clusters = [2, 3, 4, 5, 6]
inicio = time.time()
for k in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(datanp) + (k + 1) * 10])
    inicio = time.time()
    model = cluster.KMeans(n_clusters=k , init='k-means++')
    model.fit(datanp)
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"Temp: {tiempo_transcurrido} secondes")
    labels = model.labels_
    
    
    silhouette_avg=metrics.silhouette_score(datanp, labels, metric='euclidean')
    print(
        "For n_clusters =",
        k,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    
    sample_silhouette_values = silhouette_samples(datanp, labels)
    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(labels.astype(float) / k)
    ax2.scatter(
        f0, f1, marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    
    # Labeling the clusters
    centers = model.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % k,
        fontsize=14,
        fontweight="bold",
    )
fin = time.time()
tiempo_transcurrido = fin - inicio
plt.show()
print(f"Temp: {tiempo_transcurrido} secondes")