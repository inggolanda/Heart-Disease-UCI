# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:26:06 2021

@author: inggo
"""

#import libs
import pandas as pd
import seaborn as sns
sns.set()

df_heart = pd.read_csv('Datasets/heart.csv')

#filter Data dari pasien yang berusia lebih dari sama dengan 40 dan kurang dari sama dengan 55
df_f4077 = df_heart[(df_heart['age']>=40)&(df_heart['age']<=77)]

#Memisahkan Variabel

x = df_heart.iloc[:,0:13]
y = df_heart.iloc[:,-1]

# Persiapan Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
SC_X=StandardScaler()
X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)

#Proses training dengan SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
clf = SVC(kernel='rbf', random_state=0)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#evaluasi dengan confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
print(classification_report(y_test,y_pred))

score = accuracy_score(y_test,y_pred)
print(score)

