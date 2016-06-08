# -*- coding:utf8 -*- 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
from sklearn import *
from sklearn.decomposition.tests.test_nmf import random_state
np.random.seed(123)

iris = pd.read_csv("D:\\workspace\\PY\\iris.csv")
print "shape:"
print iris.shape
print "\n\n head:"
print iris.head()
print "\n\n columns:"
print iris.columns
print "\n\n index"
print iris.index
print "\n\n describe"
print iris.describe().T
print "\n\n T:"
print iris.describe()

irisK3 = cluster.KMeans(n_clusters = 3, random_state = 1)
irisFeatures = iris.ix[:,1:4]
print "\n\n"
print irisFeatures.head()
irisK3.fit(irisFeatures)
print "\n\n"    
print irisK3.labels_