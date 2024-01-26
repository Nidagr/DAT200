# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:13:55 2021

@author: Nida

CA4 
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

train_data=pd.read_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\train.csv')
X=train_data[train_data.columns[:6]]
y=train_data[train_data.columns[6]]
column_names=list(train_data.columns)
# change string values for y to be 0,1,2
le = LabelEncoder()
y = le.fit_transform(y)
le.transform(['non_habitable', 'potentially_habitable','very_habitable'])

data=X
data['target']=y
#print(train_data.isnull().sum())
"""
When executing print statement above we see that there are 543 missing values 
in feature growth_potential. We try three different ways of handling this 
situation. 
- drop rows with missing values
- KNNImputer
- IterativeImputer

Since the dataset is large it is okay to drop rows with missing values. But
we will try for all three datasets. Find best model with best hyperparameters
and an estimate of test accuracy. Then continue with the dataset that gets the 
best test accuracy. 
"""

# Drop rows with missing values
drop_na=data.dropna(axis=0)

# Impute missing values using the KNNImputer
# the dataset is large, hence I choose a large number of neighbors to impute 
# from

knn=KNNImputer(n_neighbors=20)
knn_imputed=knn.fit_transform(data)
knn_imputed=pd.DataFrame(knn_imputed,columns=column_names)

# Impute missing values using IterativeImputer
it=IterativeImputer(random_state=1)
iterative_imputed=it.fit_transform(data)
iterative_imputed=pd.DataFrame(iterative_imputed,columns=column_names)

#Impute with mean
imp_mean = SimpleImputer(strategy='mean') 
mean_imputed=imp_mean.fit_transform(data)
mean_imputed=pd.DataFrame(mean_imputed,columns=column_names)

#Impute with most frequent values
imp_freq = SimpleImputer(strategy='most_frequent')
freq_imputed=imp_freq.fit_transform(data)
freq_imputed=pd.DataFrame(freq_imputed,columns=column_names)
"""
Using the two imputers gave same values for growth_potential some places,
but different values other places. 

Time to find best models and hyperparameters to get test accuracy estimate 
for each of the three datasets. We will try for the following models:
    - Logistic Regression
    - SVC linear kernel
    - SVC rbf kernel
    - Decision trees
    - Random forests
    - KNN

Will use nested Cross-Validation to get a representative estimate of test 
accuracy. Will use pipelines when performing this. 

We have a large dataset, do not need to use a large number of folds in our
nested cross-validation loop.

Start with Logistic Regression:
    the dataset with dropped rows
"""
inner_segments = 5
outer_segments = 5

pipe_lr=make_pipeline(StandardScaler(),
                      LogisticRegression(solver = 'saga', multi_class='auto', 
                                         random_state=1))
c_val=[10**x for x in range(-4,4)]
param_grid={'logisticregression__C':c_val, 
            'logisticregression__penalty': ['l1','l2']}
#inner loop finding best hyperparameter
gs = GridSearchCV(estimator=pipe_lr, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=inner_segments,
                  n_jobs=-1)
X_na=drop_na[drop_na.columns[:6]]
y_na=drop_na[drop_na.columns[6]]
#outer loop
scores_1 = cross_val_score(gs, X_na, y_na, 
                         scoring='accuracy', cv=outer_segments)

"""
CV accuracy: 0.863 +/- 0.017

Not very high, try with the knn imputed dataset.
"""
X_knn=knn_imputed[knn_imputed.columns[:6]]
scores_2 = cross_val_score(gs, X_knn, y, 
                          scoring='accuracy', cv=outer_segments)

"""
CV accuracy: 0.855 +/- 0.022

Try with iterative imputed dataset.
"""
X_iter=iterative_imputed[iterative_imputed.columns[:6]]
scores_3 = cross_val_score(gs, X_iter, y, 
                          scoring='accuracy', cv=outer_segments)


X_mean=mean_imputed[mean_imputed.columns[:6]]
scores_4=cross_val_score(gs,X_mean,y,scoring='accuracy',cv=outer_segments)

X_freq=freq_imputed[freq_imputed.columns[:6]]
scores_5=cross_val_score(gs,X_freq,y,scoring='accuracy',cv=outer_segments)
"""
CV accuracy: 0.866 +/- 0.026

We got the best estimate for test accuracy 0.866 when using dataset where
missing values are imputed with Iterative Imputation.
"""
print('CV accuracy for drop_na: %.3f +/- %.3f' % (np.mean(scores_1),
                                      np.std(scores_1)))
print('CV accuracy for knn imputed: %.3f +/- %.3f' % (np.mean(scores_2),
                                                      np.std(scores_2)))
print('CV accuracy for iterative imputed: %.3f +/- %.3f' % (np.mean(scores_3),
                                                            np.std(scores_3)))
print('CV accuracy for mean imputed: %.3f +/- %.3f' % (np.mean(scores_4),
                                                            np.std(scores_4)))
print('CV accuracy for frequent imputed: %.3f +/- %.3f' % (np.mean(scores_5),
                                                            np.std(scores_5)))

"""
CV accuracy for drop_na: 0.863 +/- 0.017
CV accuracy for knn imputed: 0.855 +/- 0.022
CV accuracy for iterative imputed: 0.866 +/- 0.026
CV accuracy for mean imputed: 0.846 +/- 0.024
CV accuracy for frequent imputed: 0.849 +/- 0.018
"""