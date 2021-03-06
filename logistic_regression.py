# -*- coding: utf-8 -*-
"""LOGISTIC REGRESSION.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pwGiW9VgKJTmo3McPPJD4vCKlHl0XX9e

# LOGISTIC REGRESSION
### classifier which detect given flower is virginica(2) or Not.
"""

from sklearn import datasets,linear_model
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris() #loading databse
# taking omly one feature(feature no. 3) of all rows(entries).
X = iris.data[:,3:] # iris.data[:,3].reshape(150,1) or iris.data[:,3,np.newaxis]
# converting the target data into [Is this a verginica or not?] (verginica = 2 (given in dataset DESCR))
Y = (iris.target==2).astype(np.int) # flower is virginica or not # .astype used for type cast

clsfr = linear_model.LogisticRegression() # logistic regression model
clsfr.fit(X,Y) # traing model
print(clsfr.score(X,Y)) # gives score how accuratly it will predict value

example_1 = clsfr.predict([[0.6]]) 
print(example_1) # = 0 means it is not a verginica
example_2 = clsfr.predict([[2.5]])
print(example_2) # = 1 means it is a verginica

"""### Ploting graph for Visualization."""

# np.linspace() gives array of 1000 values between 0 and 3.
x_new = np.linspace(0,3,1000).reshape(-1,1) #resphae(1000[rows],1[colomn])
y_new = clsfr.predict_proba(x_new) # gives probablity of data item 
# The first index refers to the probability that the data belong to class 0,
# and the second refers to the probability that the data belong to class 1.
# These two would sum to 1.
#print(y_new)
#if you have k classes, the output would be (N,k), you would have to specify the probability of which class you want.
# just like we spicfy we want graph by probablity of class 1 (below)

plt.plot(x_new,y_new[:,1]) #y_new[:,1] taking graph of only (the probablity of data belong to class 1).

