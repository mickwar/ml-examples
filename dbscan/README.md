## [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)

DBSCAN is a density-based clustering algorithm which takes in two parameters: `eps` and `min_samples`.
Loosely, two points are part of the same cluster if they are "in the neighborhood" of or are "connected" through `min_samples` other points.
The parameter `eps` is the maximum distance between two points for them to be considered connected.
(For a more technical definition, just take a look at the Wikipedia page linked above; the algorithm is fairly straightfoward.)

Unlike other clustering algorithms, DBSCAN has the advantage of being able to cluster without specifying *a priori* the number of clusters.
There is also a designated *noise* cluster for those samples (i.e. outliers) which are too far from any discovered cluster.

### Hyper parameter selection

The choice of these parameters isn't immediately obvious, but generally, if the data set has more samples or has higher dimensions, then `min_samples` should be larger.
Selecting `min_samples` might be fairly ad hoc.

On the other hand, `eps` could be chosen using a k-nearest neighbors approach:
1. Choose `min_samples`.
2. Perform kNN where *k* is equal to `min_samples`.
3. For each point, calculate the mean distance between its neighbors.
4. Sort and plot these distances.
5. A good choice of `eps` will be where these distances begin increasing a lot, around some elbow.

### Results

The algorithm is used on two simulated data sets, each with two distinct clusters and some outliers.

#### Univariate

A mixture of three normal distributions, one of which has few samples and high variance.
Here is a plot of the data.

![](plots/uni_data.png)

We choose `min_samples` to be 10.
To find `eps`, we first perform a k-nearest neighbors clustering with *k=10*.
The following plot is the ordered kNN distances.
We let `eps` equal 0.35, as this is distance around which most samples have an inter-kNN distance.

![](plots/uni_dists.png)

With our two parameters selected, we perform the DBSCAN algorithm.
The algorithm found the two clusters and seems to also have appropriately categorized the outliers (noise).

![](plots/uni_cluster.png)

#### Bivariate

Similar to the process done with the univariate data set, but now we consider a bivariate data set.

![](plots/biv_data.png)
![](plots/biv_dists.png)
![](plots/biv_cluster.png)
