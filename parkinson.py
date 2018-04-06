#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 23:58:05 2018

@author: romanilechko
"""
# importing libs
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

N_TIMES = 20

# reading data
df = pd.read_csv("parkinsons.csv")
df = df.iloc[:, 1:]

# extracting useful feature
features = df.loc[:, df.columns != 'status'].values
labels = df.loc[:, 'status'].values

# scaling feature
scale = MinMaxScaler((-1, 1))
X = scale.fit_transform(features)

# calculating mean accuracy
def model(times, X, labels, n_components=None):
    acc = []
    
    for t in range(times):        
        if n_components != None:
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X)
            
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.14)

        
        model = XGBClassifier()
        model.fit(X_train, y_train)
    
        y_train = [round(y) for y in model.predict(X_test)]
        acc.append(accuracy_score(y_test, y_train))
        
        print("acc ", acc[-1])
    
    return sum(acc)/times

print("mean accuracy ",model(N_TIMES, X, labels))
