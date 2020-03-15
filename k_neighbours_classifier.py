# -*- coding: utf-8 -*-
"""K Neighbours Classifier

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OeXIP3xHIvBJZC_CVEqPZ5f1Vwcpe6rd

# K Neighbours Classifier

Loading Packages
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

"""Loading Dataset"""

iris = datasets.load_iris()

"""Defining Features and Label"""

#iris.DESCR
features = iris.data
label = iris.target

"""Creating and training our model"""

clsfir = KNeighborsClassifier() # creating model  ## To know how it works check notes.
clsfir.fit(features,label)      # traing model

"""testing by predicting label by giving random value"""

predict = clsfir.predict([[1,1,1,1]])
predict
