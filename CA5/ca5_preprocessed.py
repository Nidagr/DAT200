# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:02:47 2021

@author: Nida

Pipeline with regression directly on pre-processed data.

Best r2 score: 0.6786685622579214 for LASSO model.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor

#--------------------------The data-------------------------------------------
data = pd.read_pickle('train4.pkl')
data.replace('missing', np.nan, inplace=True)
# make sure data is numeric
objects=[]
for i in data.columns:
    if data.loc[:,i].dtype == object:
        objects.append(i)

data[objects] = data[objects].apply(pd.to_numeric, errors='ignore')

# test data
test = pd.read_pickle('test4.pkl')
test = test.replace('missing',np.nan)
# make sure data is numeric
ob=[]
for i in test.columns:
    if test.loc[:,i].dtype == object:
        ob.append(i)

test[ob] = test[ob].apply(pd.to_numeric, errors='ignore')

# remove the features from training and test data that had 930 missing values
liste = data.isnull().sum()
remove=[]
for i, val in enumerate(liste):
    if val > 1:
        remove.append('demo_{}'.format(i))

data = data.drop(remove,axis=1)
# we had one missing value in column demo_25, replace with mean of column
data=data.fillna(data.mean())

l = test.isnull().sum()
re=[]
for i, val in enumerate(l):
    if val > 1: 
        re.append('demo_{}'.format(i))

test = test.drop(re,axis=1)
#---------------------------------------------------------------------------

X=data.iloc[:,:-1]
y=data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


"""
We want to get an estimate of accuracy (r2 score) if we fit a linear model.
With regularization as well. That is, Ridge, least absolute shrinkage, LASSO
and Elastic Net. Perform Cross-validation.
"""

#Normal linear regression
# pipe = make_pipeline(StandardScaler(),LinearRegression())

# gs = pipe.fit(X_train, y_train)

# y_pred=gs.predict(X_test)

# print('r2 score: ', r2_score(y_test,y_pred))
# # r2 score:  0.6321191151474914

# Linear regression with LASSO

pipe = make_pipeline(StandardScaler(),Lasso())
alpha = np.arange(0.5,3.1,0.1)
param_grid = {'lasso__alpha':alpha}
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='r2', 
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train,y_train)

y_pred = gs.predict(X_test)

#print('r2 score:',r2_score(y_test,y_pred))

# This was the best estimator, save predictions of test data to file and upload
# to kaggle

reg = gs.best_estimator_
reg.fit(X,y)
predicted = reg.predict(test)
# # dataframe of predictions
id_col=[i for i in range(0,test.shape[0])]
dict_val={'Id': id_col,'Predicted': predicted}
results=pd.DataFrame(dict_val)
results=results.set_index('Id')
results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA5\\results_01.csv')

"""

r2 score: 0.6786685622579214
Got better r2 score estimate when adding values up to 3 to the parameters for 
alpha.

( Tried adding 5,10,15,20 but did not improve score.)

Makes sense as we have a lot of features and not all of them have that
much correlation with target. Probably good to make some features 0 with LASSO.
"""

# Linear Regression with Ridge

# pipe = make_pipeline(StandardScaler(),Ridge())

# param_grid={'ridge__alpha':[0.5,1,1.5,10,15,20,25,30,50,100,150,200,250]}

# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train,y_train)

# y_pred = gs.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))
"""
r2 score: 0.6784390078634808

But not realistic to include such high values of alpha, must be overfitting.
"""

# Linear regression with Elastic Net.
# pipe = make_pipeline(StandardScaler(),ElasticNet())
# param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
# param_range2 = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
# param_grid = [{'elasticnet__alpha': param_range, 
#                'elasticnet__l1_ratio': param_range2}]


# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train,y_train)

# y_pred = gs.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))
"""
r2 score: 0.6780529858653728
"""

# Decision Tree regressor

# pipe = DecisionTreeRegressor(random_state=1)
d = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# param_grid={'max_depth':d}
# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train,y_train)

# y_pred = gs.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))
"""
r2 score: 0.5219099571196113 
"""


# pipe = RandomForestRegressor(random_state=1)

# n_trees = [10,50,100,150]

# param_grid={'max_depth': d,'n_estimators':n_trees}
# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train,y_train)

# y_pred = gs.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))

"""
r2 score: 0.6952639234501086

A little better than the other scores, but beware, Random Forest is prone to
overfitting.
"""

# Polynomial model

# pipe = make_pipeline(PolynomialFeatures(),LinearRegression())

# param_grid={'polynomialfeatures__degree':[2,3,4]}

# gs = GridSearchCV(estimator=pipe,
#                   param_grid=param_grid,
#                   scoring='r2',
#                   cv=10,
#                   n_jobs=-1)

# gs = gs.fit(X_train,y_train)
# y_pred = gs.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))
"""
r2 score: -14.876237369300217
"""

# RANSAC model



# ransac = RANSACRegressor(max_trials=100,         # Number of iterations of the loop
#                           min_samples=50,         # Minimum random sample size
#                           loss='absolute_loss',   # Loss function for outliers (vertical distance)
#                           residual_threshold=5.0, # Maximum vertical distance in MAD units (test me!)
#                           stop_score=0.99,        # Stop searching if inliers score >= 0.99
#                           random_state=0)


# pipe = make_pipeline(StandardScaler(),ransac)

# # I only include the estimators that got the best scores as parameters.
# # Just give a value for alpha for the models, cannot change this.
# param_grid = {'ransacregressor__base_estimator':[LinearRegression(), 
#                         Lasso(alpha=5),Ridge(alpha=250),
#                         ElasticNet(alpha=1,l1_ratio=0.5)]}



# gs = GridSearchCV(estimator=pipe,
#                   param_grid=param_grid,
#                   scoring='r2',
#                   cv=10,
#                   n_jobs=-1)


# gs = gs.fit(X_train,y_train)
# y_pred = gs.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))
# # r2 score: 0.5178507947189692

# RANSAC with Lasso
# ransac = RANSACRegressor(base_estimator=Lasso(),
#                          max_trials=100,         # Number of iterations of the loop
#                           min_samples=50,         # Minimum random sample size
#                           loss='absolute_loss',   # Loss function for outliers (vertical distance)
#                           residual_threshold=5.0, # Maximum vertical distance in MAD units (test me!)
#                           stop_score=0.99,        # Stop searching if inliers score >= 0.99
#                           random_state=0)


# pipe = make_pipeline(StandardScaler(),ransac)


# param_grid = {'ransacregressor__base_estimator__alpha':
#               [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,2,3,4,5]}



# gs = GridSearchCV(estimator=pipe,
#                   param_grid=param_grid,
#                   scoring='r2',
#                   cv=10,
#                   n_jobs=-1)


# gs = gs.fit(X_train,y_train)
# y_pred = gs.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))
# r2 score: 0.5530629417091142

# RANSAC with Ridge

# ransac = RANSACRegressor(base_estimator=Ridge(),
#                           max_trials=100,         # Number of iterations of the loop
#                           min_samples=50,         # Minimum random sample size
#                           loss='absolute_loss',   # Loss function for outliers (vertical distance)
#                           residual_threshold=5.0, # Maximum vertical distance in MAD units (test me!)
#                           stop_score=0.99,        # Stop searching if inliers score >= 0.99
#                           random_state=0)


# pipe = make_pipeline(StandardScaler(),ransac)


# param_grid = {'ransacregressor__base_estimator__alpha':
#               [0.5,1,1.5,10,15,20,25,30,50,100,150,200,250]}



# gs = GridSearchCV(estimator=pipe,
#                   param_grid=param_grid,
#                   scoring='r2',
#                   cv=10,
#                   n_jobs=-1)


# gs = gs.fit(X_train,y_train)
# y_pred = gs.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))
# r2 score: 0.5962412062670241

# RANSAC with Elastic Net



# ransac = RANSACRegressor(base_estimator=ElasticNet(),
#                           max_trials=100,         # Number of iterations of the loop
#                           min_samples=50,         # Minimum random sample size
#                           loss='absolute_loss',   # Loss function for outliers (vertical distance)
#                           residual_threshold=5.0, # Maximum vertical distance in MAD units (test me!)
#                           stop_score=0.99,        # Stop searching if inliers score >= 0.99
#                           random_state=0)


# pipe = make_pipeline(StandardScaler(),ransac)
# param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
# param_range2 = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# param_grid = {'ransacregressor__base_estimator__alpha':param_range,
#               'ransacregressor__base_estimator__l1_ratio':param_range2}



# gs = GridSearchCV(estimator=pipe,
#                   param_grid=param_grid,
#                   scoring='r2',
#                   cv=10,
#                   n_jobs=-1)


# gs = gs.fit(X_train,y_train)
# y_pred = gs.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))
# # r2 score: 0.5577038231788025


# RANSAC with Linear Regression

# ransac = RANSACRegressor(base_estimator=LinearRegression(),
#                           max_trials=100,         # Number of iterations of the loop
#                           min_samples=50,         # Minimum random sample size
#                           loss='absolute_loss',   # Loss function for outliers (vertical distance)
#                           residual_threshold=5.0, # Maximum vertical distance in MAD units (test me!)
#                           stop_score=0.99,        # Stop searching if inliers score >= 0.99
#                           random_state=0)


# pipe = make_pipeline(StandardScaler(), ransac)

# pipe.fit(X_train,y_train)

# y_pred = pipe.predict(X_test)

# print('r2 score:',r2_score(y_test,y_pred))
# # r2 score: 0.29118820040195426






