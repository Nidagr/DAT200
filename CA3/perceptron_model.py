# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:56:50 2021

@author: Nida

CA3 Perceptron model

We have to find the best parameters for the model. In the Perceptron model 
from scikit learn there are two parameters we have to optimize:
    max_iter 
    eta0
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


train_data =pd.read_csv('train.csv',index_col=0)
X=train_data[train_data.columns[:-1]]
y=train_data[train_data.columns[8]]
eta_values=np.arange(0.5,1,0.01)
n_values =np.arange(10,101,10)
# Collect results in numpy arrays
accArr = np.zeros((len(n_values), len(eta_values)))
n_index=0
for n in n_values:
    et_index=0
    for et in eta_values:
        acc=[]
        for rs in range(1,101):
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, 
                                                            random_state=rs,stratify=y)
            sc = StandardScaler()
            sc.fit(X_train)
            # Transform (standardise) both X_train and X_test with mean and STD from
            # training data
            X_train_sc = sc.transform(X_train)
            X_test_sc = sc.transform(X_test)
            ppn = Perceptron(max_iter=n,eta0=et,random_state=1)
            ppn.fit(X_train_sc,y_train)
            y_pred = ppn.predict(X_test_sc)
            n=len(y_test)
            # find the accuracy
            accuracy = 1-(y_test!=y_pred).sum()/n
            acc.append(accuracy)
        
        avg=np.mean(acc)
        accArr[n_index,et_index]=avg
        et_index+=1
    n_index+=1
    
# Create row names for heat map
rowNames = ['{0:3.1f}'.format(n) for n in n_values]

# Create column names for heat map
colNames = ['{0:3.1f}'.format(p) for p in eta_values]

acc_df = pd.DataFrame(accArr, index=rowNames, columns=colNames)

"""
Tried first for eta in list [0.01,0.02,0.03,0.04,0.05,...,0.99]
See if accuracy better in one end and change eta values range.
The accuracy got better when eta was closer to 1, let us try a new range
starting from 0.5.

That is, we try for eta in [0.5,0.51,0.52,...,0.99]. The accuracies did not 
vary that much, think this suffices, dont need to check for a finer interval
closer to 1.

We stick with this interval for eta and try to vary the number of max_iter as 
well. max_iter in [10,20,30,40,50,60,70,80,90,100] Save accuracies in dataframe.

When examining the resulting dataframe we see that all of the accuracies lie 
really close to 0.68. The best ones had accuracy 0.681.
The perceptron model did not even reach minimum accuracy 0.75 as wanted
in this CA. Dont think further removing outliers and feature engineering
will improve it enough to be competing with the other classifiers. Think we
should move on and try other classifiers and disregard this one as not good
enough in our case. But we kind of expected this as Perceptron is not really 
widely used in modern machine learning. Mostly used for understanding the 
concepts.
"""

