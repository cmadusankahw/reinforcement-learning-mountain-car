import numpy as np

x=np.array([[-1,10,20,-8],[51,60,-7,8]])

array=np.where(x[0,:]>0)

print(array[0])
