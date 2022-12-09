# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:35:14 2022

@author: User
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

df=pd.read_csv('train.csv')
"""df=pd.read_csv('C:/Users/User/Desktop/kitap-bolum/train.csv')"""
a=df.info()
b=df.describe()
print(df.info())
print(df.describe())

df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
print(df.isnull().sum())

imp=KNNImputer(n_neighbors=3)
df.iloc[:,3:4]=imp.fit_transform(df.iloc[:,3:4])

imp2=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df.iloc[:,7:8]=imp2.fit_transform(df.iloc[:,7:8])

df.iloc[:,2]=df.iloc[:,2].replace('female','0')
df.iloc[:,2]=df.iloc[:,2].replace('male','1')

df.iloc[:,7]=df.iloc[:,7].replace('C','0')
df.iloc[:,7]=df.iloc[:,7].replace('S','1')
df.iloc[:,7]=df.iloc[:,7].replace('Q','2')

Y=df.iloc[:,0]
X=df.drop(['Survived'], axis=1)



X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=42, n_estimators=55, )
rfc.fit(X_train,Y_train)
pred_rfc=rfc.predict(X_test)
oran= accuracy_score(Y_test,pred_rfc)
print('RandomForest sınıflandırma oranı:',oran)
print(classification_report(Y_test,pred_rfc))
cm=confusion_matrix(Y_test, pred_rfc) 
print(cm)



grid = GridSearchCV(RandomForestClassifier(),
                   param_grid={'n_estimators':[i for i in range(1,60,4)],
                              'random_state':[42],
                              'min_samples_leaf':[i for i in range(1,11)]
                              },
                          
                   scoring='accuracy')
grid.fit(X_train,Y_train)
# print best parameter after tuning 
print("En iyi parametreler:",grid.best_params_) 
grid_predictions = grid.predict(X_test)

# print classification report 
print(classification_report(Y_test, grid_predictions))
"""
from joblib import dump, load
dump(rfc, 'RandomForestModel.joblib') 
"""
import pickle
dosya='RandomForestModel.pkl'
pickle.dump(grid,open(dosya,'wb'))


