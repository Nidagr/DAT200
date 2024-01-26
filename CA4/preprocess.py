# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:01:44 2021

@author: Nida

We found the best way of handling missing data (IterativeImputer) and
found the best model to choose (RandomForest). Now we need to preprocess
the data with some feature extraction (PCA,LDA,KPCA). Possible to do
LDA on our training data as we have the classes. 
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
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

#--------------PCA----------------

# # First decide on how many components we want to include
# pca=PCA(n_components=6) #we have 6 features, no more than 6 pcas
# sc=StandardScaler()
# X_std=sc.fit_transform(X)
# X_pca = pca.fit_transform(X_std)
# explVars = pca.explained_variance_ratio_
# print(explVars)
"""
From the explained variance ratio we see how much variance each principal
component explains:
    [0.4618767  0.19763922 0.13492792 0.10317267 0.05877771 0.04360578]
See that the two last pc's do not explain much variance, no point in including
these. The first foure pcs explain approximately 90% of the variance. Include
these.

Include standardscaler in pipeline because of PCA. Not because of model.
Do nested cv first to get estiamte of test accuracy, does the model
get better with pca.

Maybe let gridsearch find best number of components in pca.
"""

# pca_pipe=make_pipeline(StandardScaler(),PCA(random_state=1), 
#                         RandomForestClassifier(random_state=1))

# param_grid=[{'randomforestclassifier__criterion':['gini','entropy'],
#     'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
#     'randomforestclassifier__n_estimators':[50,75,100,125,150,175,200],
#     'pca__n_components':[1,2,3,4,5,6]}]

# inner_segments = 5
# outer_segments = 5

# pca_gs = GridSearchCV(estimator=pca_pipe, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=inner_segments,
#                   n_jobs=-1)
# scores=cross_val_score(pca_gs,X,y,scoring='accuracy', cv=outer_segments)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))
"""
When setting n_components=4 in PCA in pipeline:
    CV accuracy: 0.881 +/- 0.012
When letting number of pcs be a variable in pipeline:
    CV accuracy: 0.898 +/- 0.014
This is a little bit worse than when not doing PCA, but very close, might be
better when uploading to kaggle. Definetly try. Let us now
do some kernel PCA
"""
# #---------------Kernel PCA--------------------------------------------------
# kernel_pipe=make_pipeline(StandardScaler(),KernelPCA(random_state=1),
#                           RandomForestClassifier(random_state=1))

# gamma_val=[10**x for x in range(-3,3)]
# param_grid=[{'randomforestclassifier__criterion':['gini','entropy'],
#     'randomforestclassifier__n_estimators':[50,75,100,125,150],
#     'kernelpca__n_components':[1,2,3,4,5,6],
#     'kernelpca__kernel':['linear','rbf','poly','cosine','sigmoid',
#                           'precomputed'],'kernelpca__gamma':[0.3,0.4,0.5,0.6]}]

# inner_segments = 5
# outer_segments = 5

# kernel_gs = GridSearchCV(estimator=kernel_pipe, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=inner_segments,
#                   n_jobs=-1)
# scores=cross_val_score(kernel_gs,X,y,scoring='accuracy', cv=outer_segments)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))

"""
CV accuracy: 0.895 +/- 0.016

A little worse than normal pca, but not much.
"""

# #---------------------LDA----------------------------------
# lda_pipe=make_pipeline(StandardScaler(),LDA(),
#                        RandomForestClassifier(random_state=1))
# lda_pipe.get_params()
# param_grid={'lineardiscriminantanalysis__solver':['svd','lsqr','eigen'],
#             'randomforestclassifier__criterion':['gini','entropy'],
#             'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
#             'randomforestclassifier__n_estimators':[50,75,100,125,150]}

# inner_segments=5
# outer_segments=5
# lda_gs=GridSearchCV(estimator=lda_pipe,param_grid=param_grid,
#                     scoring='accuracy',cv=inner_segments,
#                     n_jobs=-1)
# scores=cross_val_score(lda_gs,X,y,scoring='accuracy',cv=outer_segments)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                        np.std(scores)))
"""
CV accuracy: 0.869 +/- 0.019

Worse estimate than kernel PCA and PCA. Might not concider this one. Focus on 
the other two.

Let us try and upload to kaggle when using PCA and the RandomForest. Find
best hyperparameters first.
"""
#------------------------Best model with PCA---------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
#                                                     random_state=1,stratify=y)

# pca_pipe=make_pipeline(StandardScaler(),PCA(random_state=1), 
#                         RandomForestClassifier(random_state=1))

# param_grid=[{'randomforestclassifier__criterion':['gini','entropy'],
#     'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
#     'randomforestclassifier__n_estimators':[75,100,125,150],
#     'pca__n_components':[1,2,3,4,5,6]}]


# gs = GridSearchCV(estimator=pca_pipe, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=5,
#                   n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)

"""
When splitting in training and test set we get:
0.9029773079076507
{'pca__n_components': 6, 'randomforestclassifier__criterion': 'entropy', 
 'randomforestclassifier__max_depth': None, 
 'randomforestclassifier__n_estimators': 75}

When we do not split but use all of training set (gridsearch perform cv, not
                                                  necessary to split?):
    0.9003595872423296
{'pca__n_components': 6, 'randomforestclassifier__criterion': 'gini', 
 'randomforestclassifier__max_depth': None, 
 'randomforestclassifier__n_estimators': 75}

Using the parameters in the first case (splitting in training and test) gives
predictions in csv file results_02.csv

The latter case parameters can be found in csv file results_05.csv.

Got worse score on kaggle when not splitting data.
"""
# # fit model on entire training data with best parameters
# clf = gs.best_estimator_
# clf.fit(X, y)

# predicted=clf.predict(iter_imp)

# # dataframe of predictions
# id_col=[i for i in range(0,iter_imp.shape[0])]
# dict_val={'id': id_col,'Predicted': predicted}
# results=pd.DataFrame(dict_val)
# results=results.set_index('id')
#results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\results_02.csv')

#--------------------best model with Kernel PCA ----------------------------
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
#                                                     random_state=1,stratify=y)

# kpca_pipe=make_pipeline(StandardScaler(),KernelPCA(random_state=1), 
#                         RandomForestClassifier(random_state=1))

# param_grid=[{'randomforestclassifier__criterion':['gini','entropy'],
#      'randomforestclassifier__n_estimators':[75,100,125,150],
#      'kernelpca__n_components':[1,2,3,4,5,6],
#      'kernelpca__kernel':['linear','rbf','poly','cosine','sigmoid',
#                            'precomputed'],'kernelpca__gamma':[0.3,0.4,0.5,0.6]}]

# gs = GridSearchCV(estimator=kpca_pipe, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=5,
#                   n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print(gs.best_score_)
# print(gs.best_params_)

"""
When splitting in training and test data:
0.9029773079076507
{'kernelpca__gamma': 0.3, 'kernelpca__kernel': 'linear', 
 'kernelpca__n_components': 6, 'randomforestclassifier__criterion': 'entropy', 
 'randomforestclassifier__n_estimators': 75}
When using all of training data:
    0.9023732757993745
    {'kernelpca__gamma': 0.3, 'kernelpca__kernel': 'cosine', 
     'kernelpca__n_components': 6, 'randomforestclassifier__criterion': 'gini',
     'randomforestclassifier__n_estimators': 125}
    
Parameters in first case in csv file results_03.csv, the last parameters given
in csv file results_06.csv
"""
# fit model on entire training data with best parameters
# clf = gs.best_estimator_
# clf.fit(X, y)

# predicted=clf.predict(iter_imp)

# # dataframe of predictions
# id_col=[i for i in range(0,iter_imp.shape[0])]
# dict_val={'id': id_col,'Predicted': predicted}
# results=pd.DataFrame(dict_val)
# results=results.set_index('id')
#results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\results_03.csv')










