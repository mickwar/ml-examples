import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

sns.set(palette = 'bright')


### Univariate
np.random.seed(1)
x = np.vstack((np.random.randn(500, 1),
    np.random.randn(300, 1) + 12.0,
    np.random.randn(50, 1)*10 + 6.0))

# Data plot
grid = np.linspace(np.floor(x.min()),
    np.ceil(x.max()), 
    (np.ceil(x.max()) - np.floor(x.min())) * 2 + 1)

plt.clf()
_ = plt.hist(x, color = (0,0,0), bins = grid)
plt.show(block = False)

# Use k-NN to determine optimal epsilon
minPts = 10
nbrs = NearestNeighbors(n_neighbors = minPts).fit(x)
distances, indices = nbrs.kneighbors(x)

plt.clf()
_ = plt.plot(np.sort(np.mean(distances[:,1:], 1)))
plt.show(block = False)

# Do the clustering
clustering = DBSCAN(eps=0.35, min_samples=minPts).fit(x)
clustering.labels_

# Plot the clusters
inds = np.zeros((x.shape[0], max(clustering.labels_)+1), dtype=bool)
for i in range(inds.shape[1]):
    inds[:,i] = (clustering.labels_ == i)

indq = (clustering.labels_ == -1)

pal = sns.color_palette(n_colors = inds.shape[1])
plt.clf()
for i in range(inds.shape[1]):
    _ = plt.hist(x[inds[:,i],0], color = pal[i], bins = grid)

_ = plt.hist(x[indq,0], color = (0,0,0), bins = grid)
plt.show(block = False)


### Bivariate
np.random.seed(1)
x = np.vstack((np.random.randn(500, 2)*0.5 + (-1.5, 0.5),
    np.random.randn(300, 2)*0.5 + (2.5, -0.3),
    np.random.rand(50, 2)*10 - 5))

# Plot the data
plt.clf()
_ = plt.scatter(x[:,0], x[:,1])
plt.show(block = False)

# Use k-NN to determine optimal epsilon
minPts = 10
nbrs = NearestNeighbors(n_neighbors = minPts).fit(x)
distances, indices = nbrs.kneighbors(x)

plt.clf()
_ = plt.plot(np.sort(np.mean(distances[:,1:], 1)))
plt.show(block = False)

# Do the clustering
clustering = DBSCAN(eps=0.50, min_samples=minPts).fit(x)
clustering.labels_

# Plot the clusters
inds = np.zeros((x.shape[0], max(clustering.labels_)+1), dtype=bool)
for i in range(inds.shape[1]):
    inds[:,i] = (clustering.labels_ == i)

indq = (clustering.labels_ == -1)

pal = sns.color_palette(n_colors = inds.shape[1])
plt.clf()
for i in range(inds.shape[1]):
    _ = plt.scatter(x[inds[:,i],0], x[inds[:,i],1], color = pal[i])

_ = plt.scatter(x[indq,0], x[indq,1], color = (0,0,0))
plt.show(block = False)

