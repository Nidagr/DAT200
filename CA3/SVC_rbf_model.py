# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:47:37 2021

@author: Nida

CA3 SVC kernel rbf
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train_data =pd.read_csv('train.csv',index_col=0)
X=train_data[train_data.columns[:-1]]
y=train_data[train_data.columns[8]]

c_val=np.arange(0.3,0.8,0.1)
gamma_val=np.arange(0.1,0.6,0.1)
accuracies=np.zeros((len(c_val),len(gamma_val)))

for c_ind,c in enumerate(c_val):
    for gamma_ind,gamma in enumerate(gamma_val):
        acc=[]
        for rs in range(1,101):
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                                    stratify=y,random_state=rs)
            sc = StandardScaler()
            sc.fit(X_train)
            # Transform (standardise) both X_train and X_test with mean and STD from
            # training data
            X_train_sc = sc.transform(X_train)
            X_test_sc = sc.transform(X_test)

            svc=SVC(kernel='rbf',C=c,random_state=1,gamma=gamma)
            svc.fit(X_train_sc,y_train)
            accuracy=svc.score(X_test_sc, y_test)
            acc.append(accuracy)
        accuracies[c_ind,gamma_ind]=np.mean(acc)

        
rownames=['{}'.format(round(c,1)) for c in c_val]
colnames=['{}'.format(round(g,1)) for g in gamma_val]
acc_df = pd.DataFrame(data=accuracies,index=rownames,columns=colnames)

acc_df.max().max()
"""
Tried for C=[10**(-3),...,10**(3)]
This model has accuracies a long a upside down V shape. Clearly a peak at
C=1. No question about it. Also the accuracies list shows this,
accuracy 0.7396376811594203.

Maybe try finer interval close to 1. See if the accuracy improves much with a 
more fine C value. Tried for interval [0.1,0.6,0.7,...,1.4]. Actually best 
score for 0.5, try around there next.

Tried for [0.1,0.2,...,0.5], best accuracy for 0.5 still. Seems like this is 
the best C. Not that accuracy is not very much different from when it was C=1,
new accuracy 0.7444202898550725. Not much better than for c=0.4 f.ex. But 
clearly c=0.5 gives best accuracy. No need to make even finer interval.

Also tried for different gamma values, best accuracy was
0.747971 when C=0.4 and gamma=0.1
"""
# Best model
test_data=pd.read_csv('test.csv',index_col=0)
X_test=test_data[test_data.columns[:]]
# Train model with est parameter on entire train set.
sc = StandardScaler()
sc.fit(X)
X_sc = sc.transform(X)
X_test_sc=sc.transform(X_test)

best_svc_rbf=SVC(kernel='rbf',C=0.6,random_state=1,gamma=0.01)
best_svc_rbf.fit(X_sc,y)
y_pred=best_svc_rbf.predict(X_test_sc)


# dataframe of predictions
id_col=[i for i in range(0,len(X_test))]
dict_val={'id': id_col,'Predicted': y_pred}
results=pd.DataFrame(dict_val)
results=results.set_index('id')
#results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA3\\svc_rbf_g_results.csv')

#check=pd.read_csv('svc_rbf_g_results.csv')