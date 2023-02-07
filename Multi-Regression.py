import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

def multi_var_reg(a):
  df=pd.read_csv(a)
  df.insert(0,"blank",1,True)
  print(df)
  # print(df.shape)
  x=df.iloc[:,0:-1]
  y=df.iloc[:,-1]
  # print(x.shape)
  # print("x:",x)
  # print("Y:", y)

  #multi-variable linear regression
  w=np.linalg.inv(np.dot(x.T,x)).dot(x.T).dot(y)
  # print("w:",w)
  y_pre = np.dot(x,w)
  print("y_pre", y_pre)

  # mean square error
  d1=(y-y_pre)
  print("D1:",d1)
  mse = (d1.dot(d1))/len(y)
  print("MSE: ",mse)

  # R - square calculation
  d2=(y-y.mean())
  print("D2:",d2)
  r2=1-((d1.dot(d1))/(d2.dot(d2)))
  print("R-Square: ", r2)

  # standard error
  se = sqrt((d2.dot(d2))/len(y))
  print("Standard Error: ",se)

#for 1st data-set of single variable pass dataset
multi_var_reg('C:/Users/Micro/Desktop/DL/datasets_lab4_single variable.csv')

#for 2nd data-set of two variable pass dataset
multi_var_reg('C:/Users/Micro/Desktop/DL/datasets_lab4_two variable.csv')

#for 3rd data-set of four variable
multi_var_reg('C:/Users/Micro/Desktop/DL/datasets_lab4_four variable.csv')