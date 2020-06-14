import numpy as np

x=np.random.randint(0,100,(200))
y=np.random.randint(0,100,(200))

data=np.zeros((200,2))#rows and cloumns
data[:,0]=x#all rows and 0th column
data[:,1]=y#all rows and 1th column

from sklearn.cluster import MeanShift

clsfr=MeanShift(bandwidth=10)
clsfr.fit(data)
labels=clsfr.labels_


n_cluster=len(clsfr.cluster_centers_)

color_array=['r','g','b','w','c','y','k','m']


print(n_cluster)

#print(labels)

from matplotlib import pyplot as plt

for i in range (len(labels)):
    for no in range(n_cluster):
        if(labels[i]==no):
            plt.plot(data[i][0],data[i][1],color_array[no]+str('.'))

plt.show()





