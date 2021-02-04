# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:22:02 2021

@author: T30518
"""



from sklearn.datasets import load_iris
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix



irisload = load_iris()
#print(iris.DESCR)

iris = pandas.DataFrame(irisload.data, columns=irisload.feature_names) 
iris["class"] =  irisload.target

np.amin(iris, axis=0)
np.amax(iris, axis=0)
iris.shape

# see the correlation of variable
iris_corr = iris.corr()


## visulazition
#iris.hist(bins=5, figsize=(11,8))   # variable wine_data
#plt.savefig("iris1hist.png") # 在程式所在資料夾下產生新圖檔
#plt.show()
#scatter_matrix(iris, figsize=(12,8))


# tranform as the array of variable to train algorithm  
#print(list(iris.keys()))
iris1 = iris.drop(['class'], axis=1)
X = iris.as_matrix(columns=None)


#iris1_list = iris1.values.tolist()
X3 = X[:,3].reshape(-1,1)  # petal width (花瓣寬度)

#y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0
#plt.scatter(X3,y)
#plt.show()



















