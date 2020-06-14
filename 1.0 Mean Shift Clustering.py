from sklearn import datasets

iris=datasets.load_iris()
data=iris.data


from sklearn.cluster import MeanShift
clsfr=MeanShift(bandwidth=0.85)
clsfr.fit(data)
labels=clsfr.labels_

centroids=clsfr.cluster_centers_

print(len(centroids))
print(centroids)
