import pandas as pd

import pandas_datareader as data
import datetime
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
import copy

start = pd.to_datetime('2015-01-01')
end = pd.to_datetime('today')

df = data.DataReader('FB', 'yahoo', start , end)

open=df['Open']

def data(day_one,col):
    L=[]
    n=col.shape[0]
    for k in range (day_one,n):
        M=[]
        for j in range(1,day_one+1):
            M.append(col[k-j])
        L.append(M)
    X=np.array(L)
    return X

def target (day_one,col):
    L=[]
    n=col.shape[0]
    for k in range (day_one,n):
        L.append(col[k])
    y=np.array(L)
    return y

X=data(900,open)
y= target (900,open)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
#hello
print("bonjour")
