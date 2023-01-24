# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:15:24 2023

@author: Micro
"""

import numpy as np
import pandas as pd
import statistics
import math as math
from math import pi

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv(r'/content/drive/MyDrive/Deep learning/dataset_lab2.csv')
print(df)

x = df.drop([df.columns[-1]], axis = 1)
x = x.drop([x.columns[0]], axis = 1)
y = df[df.columns[-1]]
print(y,x)
# y=df["Admission Result"]
# print(y)


print(np.unique(y))
prior_prob={}
for i in np.unique(y):
  i_cout=0
  for j in y:
    if i ==j:
      i_cout=i_cout+1
  prior_prob[i]= i_cout/len(y)

print(prior_prob)


yes_indices = []
no_indices = []
for i in range(len(y)):
  # y.any('YES')
  if (y[i]=='YES'):
    yes_indices.append(i)
  else:
    no_indices.append(i)    

print(yes_indices)
print(no_indices)



def mean(x1):
  sum=0
  for i in x1:
    sum=(sum+i)
  return (sum/len(x1))


def sd(x1):
  return (statistics.stdev(x1))


def liklyhood(x1,x2):
  prob={}  
  a=0
  if x2=='YES':
    ind = yes_indices
  elif x2=='NO':
    ind = no_indices

  for i in x.columns:
    prob1=0
    xi = ([x[i][k] for k in ind])
    if (type(xi[0])!=str):
      prob1+=(1/(((2*pi)**0.5)*sd(xi)))*(math.exp((-((x1[a][0])-mean(xi))**2)/(2*(sd(xi))**2)))
      if(x1[a][1]==1):
        prob1=1-prob1      
      prob[i]=prob1
      a=a+1
    elif(type(xi[0])==str):
      count=0
      if x1[a][0]=='Yes':
        for j in xi:
          if j=='Yes':
            count+=1
      elif x1[a][0]=='No':
        for j in xi:
          if j=='No':
            count+=1
      prob[i] = count/len(xi)
  return prob


x1=[(65,0),(65,0),(65,0),(70,1),(400,1),('No',1)]
liklyhood_prob=liklyhood(x1,"YES")

lik_prob_yes=1
for i in liklyhood_prob.values():
  lik_prob_yes=lik_prob_yes*i

final_prob= (lik_prob_yes*prior_prob['YES'])
print("Probability: ",final_prob)



x1=[(80,1),(75,1),(65,1),(70,0),(400,0),('Yes',1)]
liklyhood_prob=liklyhood(x1,"NO")


lik_prob_no=1
for i in liklyhood_prob.values():
  lik_prob_no=lik_prob_no*i

final_prob= (lik_prob_no*prior_prob['NO'])
print("Probability: ",final_prob)


x1=[(80,1),(75,1),(65,1),(60,0),(400,0),('No',1)]
liklyhood_prob=liklyhood(x1,"YES")

lik_prob_yes=1
for i in liklyhood_prob.values():
  lik_prob_yes=lik_prob_yes*i

final_prob= (lik_prob_yes*prior_prob['YES'])
print("Probability: ",final_prob)


test = pd.read_csv(r'/content/drive/MyDrive/Deep learning/testset_lab2.csv')
print(test)

x_test = test.drop([test.columns[-1]], axis = 1)
x_test = x_test.drop([x_test.columns[0]], axis = 1)
y_test = test[test.columns[-1]]
print(y_test,x_test)


print(x_test.columns)

prediction=[]
for i in range(len(x_test['Class 10'])):
  x1=[(x_test['Class 10'][i],0),(x_test['Class 12'][i],0),(x_test['UG'][i],0),
      (x_test['PG'][i],0),(x_test['GATE Score'][i],0),(x_test['Work Exp'][i],0)]


  #for yes
  liklyhood_prob=liklyhood(x1,"YES")
  lik_prob_yes=1
  for i in liklyhood_prob.values():
    lik_prob_yes=lik_prob_yes*i
  final_prob_yes= (lik_prob_yes*prior_prob['YES'])


  #for no
  liklyhood_prob=liklyhood(x1,"NO")
  lik_prob_no=1
  for i in liklyhood_prob.values():
    lik_prob_no=lik_prob_no*i
  final_prob_no= (lik_prob_no*prior_prob['NO'])



  # Finding output whos prob is more
  if(final_prob_yes>final_prob_no):
    prediction.append('YES')  
  else:
    prediction.append('NO')



prediction


accuracy = (sum(1 for x_i,y_i in zip(y_test,prediction) if x_i == y_i) / float(len(prediction)))*100

print("accuracy ", accuracy,"%")


import numpy as np
from sklearn.metrics import precision_recall_fscore_support




precision_recall_fscore_support(y_test, prediction, average='macro')







