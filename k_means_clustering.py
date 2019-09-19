# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
10.5.1

Kasper Kronborg Larsen
"""

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.cluster import KMeans

#Laver array af 50x2 random datapunkter
np.random.seed(1000)
X = np.random.randn(50,2)

#Forskellig drift i hver kolonne 
X[0:25, 0] = X[0:25, 0] + 3
X[0:25, 1] = X[0:25, 1] - 4

#Plotter random datapunkter
f, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:,0], X[:,1], s=30) 
ax.set_xlabel('X0')
ax.set_ylabel('X1')

#%% K-means clustering med k = 2
k_2 = KMeans(n_clusters = 2, random_state = 1000).fit(X)
print(k_2.labels_)

##Opdeler data baseret på k-means clustering-resultatet
plt.figure(figsize=(6,5))
plt.scatter(X[:,0],X[:,1],s=30,c=k_2.labels_,cmap=plt.cm.bwr)
plt.scatter(k_2.cluster_centers_[:,0],
            k_2.cluster_centers_[:,1],
            marker='*',s=150,color='black',
            label='Cluster centers')
plt.legend(loc='best')
plt.xlabel('X0')
plt.ylabel('X1')

# Cluster means
print(k_2.cluster_centers_)

#%% K-means clustering med k = 3
k_3 = KMeans(n_clusters = 3, random_state = 1000).fit(X)
print(k_3.labels_)

plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], 
            s=50, 
            c=k_3.labels_, 
            cmap=plt.cm.prism) 
plt.scatter(k_3.cluster_centers_[:, 0], 
            k_3.cluster_centers_[:, 1], 
            marker='*', 
            s=150,
            color='black', 
            label='Cluster centers')
plt.legend(loc='best')
plt.xlabel('X0')
plt.ylabel('X1')

# Cluster means
print(k_3.cluster_centers_)

#%% Multiple inital cluster assignments

for n in 1,20:
    kmeans = KMeans(n_clusters = 3, n_init = n, random_state = 11).fit(X)
    #kmeans.inertia_
    
    plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], 
            s=50, 
            c=kmeans.labels_, 
            cmap=plt.cm.prism) 
    plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            marker='*', 
            s=150,
            color='black', 
            label='Cluster centers')
    plt.legend(loc='best')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title('Cluster with k=3 and n={}, inertia = {}'.format(n,kmeans.inertia_))
    print('K = 3', 'n =', n, 'Inertia=', kmeans.inertia_)
    print(kmeans.cluster_centers_)




