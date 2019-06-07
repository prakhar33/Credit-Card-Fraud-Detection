#Data preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#Importing the dataset
dataset=pd.read_csv('creditcard.csv')
dataset=dataset.sample(frac=0.1,random_state=1)
fraud=dataset[dataset['Class']==1]
valid=dataset[dataset['Class']==0]
outlier_frac=len(fraud)/float(len(valid))
corrmat=dataset.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)
plt.show()

columns=dataset.columns.tolist()
columns=[c for c in columns if c not in ['Class']]
target='Class'
X=dataset[columns]
y=dataset[target]

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor  

#define the outlier detection methods
classifiers={'Isolation Forest':IsolationForest(max_samples=len(X),contamination=outlier_frac,random_state=1),
             'Local Outlier Factor':LocalOutlierFactor(n_neighbors=20,contamination=outlier_frac)}
 


for i,(clf_name,clf) in enumerate(classifiers.items()):
    
    if clf_name=='Local Outlier Factor':
        y_pred=clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
        
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)  
        
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1

    n_errors=(y_pred!=y).sum()

    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))  

      