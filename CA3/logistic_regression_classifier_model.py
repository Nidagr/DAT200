# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:52:20 2021

@author: Nida

CA3 Logistic regression model

The parameters we need to optimize are:
    C (positive float)
    
LogisticRegression(C=c,random_state=1,solver='liblinear',multi_class='auto')
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


train_data =pd.read_csv('train.csv',index_col=0)
X=train_data[train_data.columns[:-1]]
y=train_data[train_data.columns[8]]

c_val=[10**x for x in range(-5,11)]
mean_accuracies=[]
for c in c_val:
    acc=[]
    for rs in range(1,101):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                                    random_state=rs,stratify=y)
        sc = StandardScaler()
        sc.fit(X_train)
        # Transform (standardise) both X_train and X_test with mean and STD from
        # training data
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)
        logreg=LogisticRegression(C=c,solver='liblinear',multi_class='auto',
                                  random_state=1)
        logreg.fit(X_train_sc, y_train)
        accuracy=logreg.score(X_test_sc,y_test)
        acc.append(accuracy)
    mean_accuracies.append(np.mean(acc))


accuracies = {'test acc':mean_accuracies}
acc_df = pd.DataFrame(data=accuracies)

# Add column holding strings indicating value of C. Needed for xticks lables
acc_df['C'] = ['10**{0}'.format(c) for c in range(-5,11)]

# Plot columns acc train and acc test. Define xticks lables
ax = acc_df.plot(xticks=acc_df.index, rot=45)
ax.set_xticklabels(acc_df.C)
plt.grid()
plt.show()

    
"""
Tried first for C in [10**(-10),...,10**(10)]. The accuracies becomes greater
than 0.76 when C is greater than or equal to 0.1. The last accuracies are about 
the same, but let us try to change C to be in the interval [10**(-1),10**(20)]
to see if even greater C is better.

The accuracies did not change, stayed constantly the same. Actually seem
like the best accuracy was for C=10**3 and larger. We can plot the accuracies 
to make it more clear.

Plot for C in [10**(-5),...,10**(10)]

When examining the plot we actually see that the test accuracy reaches its peak
when C=1. After that the curve falttens out. If there is an accuracy better
later on the difference is very small and we cannot see it. The best parameter
to use in our model is C=1.

The best accuracy: 0.7628260869565219

To further improve the model one could remove outliers and do some feature 
engineering. But let us first try to do similar for the other
models, and then use the best model to remove outliers and do feature 
engineering to improve accuracy later on.
"""
test_data=pd.read_csv('test.csv',index_col=0)
X_test=test_data[test_data.columns[:]]
# Train model with est parameter on entire train set.
sc = StandardScaler()
sc.fit(X)
X_sc = sc.transform(X)

best_logreg=LogisticRegression(C=0.1,solver='liblinear',multi_class='auto',
                                  random_state=1,penalty='l1')
best_logreg.fit(X_sc,y)
sc.fit(X_test)
X_test_sc=sc.transform(X_test)
y_pred=best_logreg.predict(X_test_sc)

# dataframe of predictions
id_col=[i for i in range(0,len(X_test))]
dict_val={'id': id_col,'Predicted': y_pred}
results=pd.DataFrame(dict_val)
results=results.set_index('id')
results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA3\\logreg_results.csv')

check=pd.read_csv('logreg_results.csv')

"""

Submitted for logreg C=0.1 without and with l1 penalization. Did not change 
the score to use l1 penalization.
"""