# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:21:13 2023

@author: Micro
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv(r'/content/drive/MyDrive/Deep learning/dataset_lab2.csv')
print(df)

x = df.drop([df.columns[-1]], axis = 1)
# x = x.drop([x.columns[-1]], axis = 1)
x = x.drop([x.columns[0]], axis = 1)
y = df[df.columns[-1]]

a = x['Work Exp']
b=0
for i in a:
  if i =='Yes':
    x['Work Exp'][b]=1
  else:
    x['Work Exp'][b]=0
  b+=1
  
  
features=x.columns
features

dtree = DecisionTreeClassifier()
dtree = dtree.fit(x, y)

tree.plot_tree(dtree, feature_names=features)


print(dtree.predict([[67, 72, 73, 82, 456,0]]))

test = pd.read_csv(r'/content/drive/MyDrive/Deep learning/testset_lab2.csv')
print(test)


x_test = test.drop([test.columns[-1]], axis = 1)
x_test = x_test.drop([x_test.columns[0]], axis = 1)
y_test = test[test.columns[-1]]
print(y_test,x_test)


a = x_test['Work Exp']
b=0
for i in a:
  if i =='Yes':
    x_test['Work Exp'][b]=1
  else:
    x_test['Work Exp'][b]=0
  b+=1


features=x_test.columns
features


a1=[]

for i in range(len(x_test['Class 10'])):
  a = x_test['Class 10'][i]
  b= x_test['Class 12'][i]
  c= x_test['UG'][i]
  d=x_test['PG'][i]
  e=x_test['GATE Score'][i]
  f=x_test['Work Exp'][i]
  print(dtree.predict([[a, b, c, d, e,f]]))
  a1.append((dtree.predict([[a, b, c, d, e,f]]))[0])


accuracy = (sum(1 for x_i,y_i in zip(y_test,a1) if x_i == y_i) / float(len(a1)))*100

print("accuracy ", accuracy,"%")



import numpy as np
from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(y_test, a1, average='macro')




