# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 08:08:41 2021

@author: Nida

CA4

We use the dataset where missing values are imputed with IterativeImputer 
as this performed best.

Got best accuracy for 125 estimators, maybe try and increase the number of 
estimators to see if it gets better accuracy according to nested CV.
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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
X_iter=iterative_imputed[iterative_imputed.columns[:6]]

# #---------------- Find test accuracy estimate ------------- 
# RF_pipe = make_pipeline(RandomForestClassifier(random_state=1))
# param_grid=[{'randomforestclassifier__criterion':['gini','entropy'],
#     'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
#     'randomforestclassifier__n_estimators':[175,200,225,250]}]

# inner_segments = 5
# outer_segments = 5

# gs = GridSearchCV(estimator=RF_pipe, 
#                   param_grid=param_grid, 
#                   scoring='accuracy', 
#                   cv=inner_segments,
#                   n_jobs=-1)

# X_iter=iterative_imputed[iterative_imputed.columns[:6]]
# scores = cross_val_score(gs, X_iter, y, 
#                           scoring='accuracy', cv=outer_segments)

# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))
"""
n_estimators=25,50,75,100,125:
CV accuracy: 0.901 +/- 0.018

n_estimators=125,150,175,200:
    CV accuracy: 0.903 +/- 0.016
    
n_estimators=175,200,225,250:
    CV accuracy: 0.902 +/- 0.016
    

To find the best hyperparameters we cannot use nested CV. Try the old fashioned
way.

I think we do not need to split in training and test data as the gridsearch
performs CV. However they asked us to return performance metrics, so we need
to split in training and test data for this purpose??
"""
X_train, X_test, y_train, y_test = train_test_split(X_iter,y,test_size=0.3,
                                                    random_state=1,stratify=y)

RF_pipe = make_pipeline(RandomForestClassifier(random_state=1))


param_grid=[{'randomforestclassifier__criterion':['gini','entropy'],
    'randomforestclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, None],
    'randomforestclassifier__n_estimators':[50,75,100,125,150,175,200,250]}]

gs = GridSearchCV(estimator=RF_pipe, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=5,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
"""
When splitting training data in train and test split:
0.9029855416131476
{'randomforestclassifier__criterion': 'entropy', 
 'randomforestclassifier__max_depth': None, 
 'randomforestclassifier__n_estimators': 125}

When we do not split the training data we got slightly different output:
    0.9039797015021858
{'randomforestclassifier__criterion': 'entropy', 
 'randomforestclassifier__max_depth': None, 
 'randomforestclassifier__n_estimators': 75}

Uploaded both choices to kaggle. Got best for the first version, 
n_estimators=125. Seems like a good idea to split in training and test set.

Confusion matrix:
"""
y_pred=gs.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.show()

"""
Fit the best model on all of the training data and predict labels for 
test data
""" 
# test = pd.read_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\test.csv')
# X_p = test[test.columns[:6]]
# # missing values in test data, impute
# iter_imp=it.fit_transform(X_p)
# iter_imp=pd.DataFrame(iter_imp,columns=column_names[:-1])


# clf = gs.best_estimator_
# clf.fit(X_iter, y)

# predicted=clf.predict(iter_imp)

# # dataframe of predictions
# id_col=[i for i in range(0,iter_imp.shape[0])]
# dict_val={'id': id_col,'Predicted': predicted}
# results=pd.DataFrame(dict_val)
# results=results.set_index('id')
#results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA4\\results_01.csv')













