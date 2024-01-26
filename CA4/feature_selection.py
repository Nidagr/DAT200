# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:52:57 2021

@author: Nida

CA4

Best model with PCA and RandomForest classifier. Now try to even better the
model. Maybe we can include feature selection before PCA.

"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


# get training data
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

# Impute missing values using IterativeImputer
it=IterativeImputer(random_state=1)
iterative_imputed=it.fit_transform(data)
iterative_imputed=pd.DataFrame(iterative_imputed,columns=column_names)
X=iterative_imputed[iterative_imputed.columns[:6]]

# impute test data
test = pd.read_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\test.csv')
X_p = test[test.columns[:6]]
# missing values in test data, impute
iter_imp=it.fit_transform(X_p)
iter_imp=pd.DataFrame(iter_imp,columns=column_names[:-1])

# fs_pipe=make_pipeline(StandardScaler(), 
#                       SelectFromModel(estimator=LogisticRegression()),
#                       PCA(random_state=1),
#                       RandomForestClassifier(random_state=1))

# param_grid=[{'selectfrommodel__threshold':['0.5*mean','0.75*mean','mean',
#                                            '1.25*mean'],
#              'randomforestclassifier__criterion':['gini','entropy'],
#     'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
#     'randomforestclassifier__n_estimators':[50,75,100,125,150],
#     'pca__n_components':[1,2,3,4,5,6]}]

# # Nested CV to get test accuracy estimate
# inner_segments = 5
# outer_segments = 5

# pca_gs = GridSearchCV(estimator=fs_pipe, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=inner_segments,
#                   n_jobs=-1)
# scores=cross_val_score(pca_gs,X,y,scoring='accuracy', cv=outer_segments)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))

"""
CV accuracy: 0.867 +/- 0.014

This estimate is worse than when not doing feature selection. Maybe try 
removing features but not performing pca.
"""

# fs_pipe=make_pipeline(StandardScaler(), 
#                       SelectFromModel(estimator=LogisticRegression()),
#                       RandomForestClassifier(random_state=1))

# param_grid=[{'selectfrommodel__threshold':['0.5*mean','0.75*mean','mean',
#                                            '1.25*mean'],
#              'randomforestclassifier__criterion':['gini','entropy'],
#     'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
#     'randomforestclassifier__n_estimators':[50,75,100,125,150]}]

# inner_segments = 5
# outer_segments = 5

# pca_gs = GridSearchCV(estimator=fs_pipe, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=inner_segments,
#                   n_jobs=-1)
# scores=cross_val_score(pca_gs,X,y,scoring='accuracy', cv=outer_segments)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))

"""
CV accuracy: 0.867 +/- 0.018

The exact same score. Seems like using feature selection does not help the 
model. 

However it could be that kaggle tells us something else, as there is not much
difference from the case where we only use pca beforehand. Maybe try and upload
to kaggle and see.
"""
fs_pipe=make_pipeline(StandardScaler(), 
                      SelectFromModel(estimator=LogisticRegression()),
                      PCA(random_state=1),
                      RandomForestClassifier(random_state=1))

param_grid=[{'selectfrommodel__threshold':['0.5*mean','0.75*mean','mean',
                                            '1.25*mean'],
              'randomforestclassifier__criterion':['gini','entropy'],
    'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
    'randomforestclassifier__n_estimators':[50,75,100,125,150],
    'pca__n_components':[1,2,3,4,5,6]}]



pca_gs = GridSearchCV(estimator=fs_pipe, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=5,
                  n_jobs=-1)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                    random_state=1,stratify=y)

gs = pca_gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
"""
0.8748542634127062
{'pca__n_components': 3, 'randomforestclassifier__criterion': 'gini', 
 'randomforestclassifier__max_depth': 6, 
 'randomforestclassifier__n_estimators': 125, 'selectfrommodel__threshold': 
     '0.5*mean'}
"""
# fit model on entire training data with best parameters
clf = gs.best_estimator_
clf.fit(X, y)

predicted=clf.predict(iter_imp)

# dataframe of predictions
id_col=[i for i in range(0,iter_imp.shape[0])]
dict_val={'id': id_col,'Predicted': predicted}
results=pd.DataFrame(dict_val)
results=results.set_index('id')
results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\results_07.csv')

"""
When uploading to kaggle I got a score of 0.79819, the worst one yet. Clearly
the nested CV was correct in predicting low accuracy for the feature selection
case. Better to not do this. As the feature selection without pca got same
test accuracy estimate in nested cv I see no point in uploading this one as 
well. Let this be.
"""