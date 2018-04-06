#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 23:58:05 2018

@author: romanilechko
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("parkinsons.csv")
df = df.iloc[:, 1:]

features = df.loc[:, df.columns != 'status'].values
labels = df.loc[:, 'status'].values

scale = MinMaxScaler((-1, 1))
X = scale.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1)

model = XGBClassifier()
model.fit(X_train, y_train)

y_train = [round(yhat) for yhat in model.predict(X_test)]
print(accuracy_score(y_test, y_train))
