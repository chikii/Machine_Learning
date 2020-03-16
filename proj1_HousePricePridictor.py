#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Import data

# In[2]:


housing = pd.read_csv('DATA/housing.csv')
housing.head() #checking : data is imported properly


# # <center> Analysis of DATA </center>

# In[ ]:


housing.info() #getting information about data


# **found :** missing data in RM feature.

# In[ ]:


housing.describe() # to get overview od whole data.


# **found :** CHAS has more than 75% of 0's. so in order to properly distribute data in train,test set.
#             we have to do stratified shuffeling.

# In[ ]:


housing['CHAS'].value_counts()


# while spliting if no 1's goes in train data then our model will not know that CHAS even has 1 value too.
# so proper shuffeling is very important to build good model.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
#col_study = ['RM','LSTAT','B','MEDV']
#housing[col_study].hist(bins = 50 , figsize = (10,5)) # bins is something #zoom more exact view
#plt.show()


# ![image.png](attachment:image.png)

# In[ ]:


plt.scatter(housing['RM'],housing['MEDV'],alpha=0.5)


# for better analysing , using seaborn

# In[ ]:


#col_study = ['RM','LSTAT','B','MEDV']
#sns.pairplot(housing[col_study],height=2)
#plt.show()


# ![image.png](attachment:image.png)

# **found :** here we get that RM and LSTAT has linear regression. i.e. on increasing RM our price is also increasing. and on increasing LSTAT our price is decreasing. and B is not affecting price so much.

# ## Looking for CORRELATION

# In[ ]:


corr_matrix = housing.corr()
corr_matrix # print relation to everyone vs everyone
corr_matrix['MEDV'].sort_values(ascending = False)


# <h4><center>correalation values are from -1 to 1 </center> </h4>
# 1 means highly positively related to MEDV (because we are checking correlation pf MEDV) 
# which means **propotional to MEDV.** <br>
# -1 means highly negativly related to MEDV.
# which means **inversly-propotional to MEDV.** <br>

# ### Trying a new feature 

# In[ ]:


tax_per_rm = housing['RM']/housing['TAX']
housing['TPR'] = tax_per_rm
housing.head()


# In[ ]:


# now lets check how much our new attribute is correlated to our target. 
housing.corr()['MEDV'].sort_values(ascending = False) #ascending = false to sort in descending order.


# oh!! Great our new attribute score very good.
# it is correlated next to RM. <br>
# **In this way, we can try out Attribute combinations.**

# In[ ]:


#lets check the scatter plot of TRM
plt.scatter(housing['TPR'],housing['MEDV'])
plt.show()


# it gives good linear relation with price. <br>
# BUT there are some outlier. (see left upper corner)

# In[ ]:


#for now droping the new feature. To work on actual data.
housing = housing.drop('TPR',axis = 1)
housing.head()


# ## Outcomes of Analysis
# **1.** we have missing values in RM. <br>
# **2.** Stratisfied shuffeling is to be done w.r.t CHAS

# ## <center>...Analysis Complete...</center>

# # Split DATA  into training and testing data
# here we have to remember that stratisfied spliting is to be done w.r.t CHAS 

# In[12]:


from sklearn.model_selection import StratifiedShuffleSplit
sssplit = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2)#random_state = 0)
for train_indices,test_indices in sssplit.split(housing,housing['CHAS']):
    train_data = housing.loc[train_indices]
    test_data = housing.loc[test_indices]
#train_data


# In[33]:


test_data


# In[13]:


# To check
print(train_data['CHAS'].value_counts())
print(test_data['CHAS'].value_counts())


# 376/28 = 13.392857142857142 <br>
# 95/7   = 13.392857142857142 <br>
# Both in train and test data equal proptional of chas is distributed as it's propotional in original data

# In[14]:


# All the stuff we will apply on train_data and hide the test data from machine
housing = train_data.drop('MEDV',axis=1)
housing_labels = train_data['MEDV'].copy()


# # Taking care of Missing Values.

# ##### To take care of missing attributes, you have three options:
# #####     1. Get rid of the missing data points (delete that row)
# #####    2. Get rid of the whole attribute (drop that attribute . if it is not much correlated to target.)
# #####     3. Set the value to some value(0(a constant value), mean or median or max_occured)

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean')
imputer.fit(housing) # calcute value to impute acc. to strategy on basis of data passed in it.
print(imputer.statistics_)


# values what imputer is computed for particular column. after fit()

# In[ ]:


housing.describe()


# here RM has 401 values

# In[ ]:


#lets fill the missing values in Actual Data and transform data in numpy array.
# we always transform data before providing it to model
housing_nump_tr = imputer.transform(housing) # this convert data into numpyarray
housing_trnsfm = pd.DataFrame(housing_nump_tr,columns = housing.columns)
housing_trnsfm.describe()


# here we can see RM has now 404 values means all the missing values is now imputed b imputer

# ##### now for everyy data we have to fit in model first we have to transform it 
# ## simple way of doing it is CREATING a PIPELINE

# In[15]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# In[16]:


my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('sclaer', StandardScaler())
])


# Primarily, two types of feature scaling methods:
# 1. Min-max scaling (Normalization)
#     (value - min)/(max - min)
#     Sklearn provides a class called MinMaxScaler for this
#     
# 2. Standardization
#     (value - mean)/std
#     Sklearn provides a class called StandardScaler for this
# 

# In[17]:


housing_nump_trnsfm = my_pipeline.fit_transform(housing)
housing_nump_trnsfm.shape


# # <center>Selecting a Model</center>

# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_nump_trnsfm,housing_labels)


# ## Before giving the test data to model, we check the which model is best by evaluting performance measure by taking some data from train data.

# In[19]:


some_data = housing.iloc[-5:]
some_labels = housing_labels.iloc[-5:]


# In[20]:


#transform the data before giving it to model
prepared_data = my_pipeline.transform(some_data)


# In[21]:


predict_data = model.predict(prepared_data)
print('original : ',list(some_labels))
predict_data


# #### Evaluting Model

# In[22]:


#Error measuring - cost function - Root mean square value
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(some_labels,predict_data)
rmse = np.sqrt(mse)
rmse


# 0.0 because model OVERFIT the data

# ### Better Evalution Technique : Cross Validation

# In[23]:


from sklearn.model_selection import cross_val_score
scores= cross_val_score(model,housing_nump_trnsfm,housing_labels,scoring='neg_mean_squared_error',cv=10)#cv =1 2 3 4 5 6 7 8 9 10
rmse = np.sqrt(-scores)
rmse


# In[24]:


def print_scores(scores):
    print('Scores : ',scores)
    print('Mean : ', scores.mean())
    print('Standard Deviation : ', scores.std())
print_scores(rmse)


# Scores of Different Model:
# 1. **LinearRegression :**
# Scores :  [4.19827405 4.29449888 5.10631166 3.83030298 5.36400513 4.38752225
#  7.45454267 5.48662579 4.14297624 6.06370639] <br>
# Mean :  5.032876604032856 <br>
# Standard Deviation :  1.0556682722554054 <br>
# 2. **DecisionTree :**
# Scores :  [3.88985547 5.70793201 5.28132929 4.0948808  3.43722272 4.54221862
#  8.22119213 3.82896853 3.39448081 4.6219855 ] <br>
# Mean :  4.702006588210912 <br>
# Standard Deviation :  1.373455192706483<br>
# 
# 3. **RandomForest :**
# Scores :  [2.82133218 2.90556382 4.61958137 2.48725064 3.39808365 2.77115465 
#  4.68374218 3.32223701 2.84569201 3.22914651] <br>
# Mean :  3.3083784026888687 <br>
# Standard Deviation :  0.7212356076172443 <br>

# # Saving The MODEL

# In[25]:


from joblib import dump,load
dump(model,'HousePricePredicter.joblib')


# # Testing The Model

# In[35]:


X_test = test_data.drop('MEDV',axis=1)
Y_test = test_data['MEDV'].copy()
X_prepared = my_pipeline.transform(X_test)
final_prediction = model.predict(X_prepared)
final_mse = mean_squared_error(Y_test,final_prediction)
final_rmse = np.sqrt(mse)


# In[36]:


final_rmse


# # Using the MODEL

# In[34]:


from joblib import dump,load
import numpy as np
model_use = load('HousePricePredicter.joblib')


# In[32]:


y = my_pipeline.transform(X_prepared[[0]])
y
print(Y_test)


# In[30]:


predict = model_use.predict([[-0.44228927, -0.4898311 , -1.37640684, -0.27288841, -0.34321545,
         0.36581973, -0.33092752,  1.20235683, -1.0016859 ,  0.05733231,
        -1.21003475,  0.38110555, -0.57309194]])
predict


# In[ ]:




