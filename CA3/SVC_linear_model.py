# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:42:30 2021

@author: Nida

CA3 Support Vector classifire with SVC kernel 'linear'
We need to find the best parameter for this model, the parameters
to optimize are:
    C (>0)

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

train_data =pd.read_csv('train.csv',index_col=0)
X=train_data[train_data.columns[:-1]]
y=train_data[train_data.columns[8]]

c_val=[10**x for x in range(-3,4)]
mean_accuracies=[]
for c in c_val:
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
        
        svc=SVC(kernel='linear',C=c,random_state=1)
        svc.fit(X_train_sc,y_train)
        accuracy=svc.score(X_test_sc, y_test)
        acc.append(accuracy)
    mean_accuracies.append(np.mean(acc))
        
        
# Construct pandas dataframe from lists with accuracies
accuracies = {'test acc':mean_accuracies}
acc_df = pd.DataFrame(data=accuracies)

# Add column holding strings indicating value of C. Needed for xticks lables
acc_df['C'] = ['10**{0}'.format(c) for c in range(-3,4)]

# Plot columns acc train and acc test. Define xticks lables
ax = acc_df.plot(xticks=acc_df.index, rot=45)
ax.set_xticklabels(acc_df.C)

# Set axis lables
ax.set_title('SVC')
ax.set_xlabel('C')
ax.set_ylabel('Classification accuracy')

        
        
        
        
"""
We can see that the accuracy improves when C gets larger but stabilizes really 
quick. Already when C=10^(-2) we reach the accuracy peak and the performance 
stabilizes. If we examine the array of accuracies closely there is a very small 
improvement in accuracy for sligthly larger C. But so small that we cannot see 
it in the plot. Let us decide that the best C value is 10^(-2). Looking at a 
finer smaller interval around this C value could have been interesting. But the
accuracies did not change much in this case so the difference in a finer 
interval would probably not be much better.

[0.6521739130434784,
 0.7637681159420291,
 0.7645652173913043,
 0.7647101449275362,
 0.7642753623188406,
 0.7641304347826088,
 0.7640579710144928]

Actually best accuracy for C=10**(0)=1. But enough difference for us to choose
this C?


"""

# best model c=10^(-2)
test_data=pd.read_csv('test.csv',index_col=0)
X_test=test_data[test_data.columns[:]]
# Train model with est parameter on entire train set.
sc = StandardScaler()
sc.fit(X)
X_sc = sc.transform(X)
X_test_sc=sc.transform(X_test)

best_svc=SVC(kernel='linear',C=10**(-2),random_state=1)
best_svc.fit(X_sc,y)
y_pred=best_svc.predict(X_test_sc)


# dataframe of predictions
id_col=[i for i in range(0,len(X_test))]
dict_val={'id': id_col,'Predicted': y_pred}
results=pd.DataFrame(dict_val)
results=results.set_index('id')
results.to_csv('C:\\Users\\Nida\\NMBU\\DAT200\\CA3\\svc_linear_results.csv')

check=pd.read_csv('svc_linear_results.csv')