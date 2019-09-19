# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
10.5.2

Kasper Kronborg Larsen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

np.random.seed(1000)
X = np.random.randn(50,2) # 50 data points (x and y values)
X[0:25, 0] = X[0:25, 0] + 3 # shift of first 25 x-values
X[0:25, 1] = X[0:25, 1] - 4 # shift of first 25 y-values


# Different linkages - complete, average and single
hc_complete = linkage(X, "complete")
hc_average = linkage(X, "average")
hc_single = linkage(X, "single")

# Plot complete linkage
plt.figure(figsize=(23, 10))
plt.title('Hierarchical clustering dendrogram - complete linkage')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    hc_complete,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Plot average linkage
plt.figure(figsize=(23, 10))
plt.title('Hierarchical clustering dendrogram - average linkage')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    hc_average,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Plot single linkage
plt.figure(figsize=(23, 10))
plt.title('Hierarchical clustering dendrogram - single linkage')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    hc_single,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

from scipy.cluster.hierarchy import cut_tree
print("Complete: ", cut_tree(hc_complete, n_clusters = 2).T) 
print("Average: ", cut_tree(hc_average, n_clusters = 2).T) 
print("Single: ", cut_tree(hc_single, n_clusters = 2).T) 
