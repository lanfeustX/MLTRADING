#!/usr/bin/env python
# coding: utf-8

# # Preparation des données

# In[90]:


import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import MinMaxScaler #pour standardiser les données 


# In[92]:


df = pd.read_csv("FB_tiingo .csv")


# In[93]:


# Répare une incompatibilité entre scipy 1.0 et statsmodels 0.8. -> Dans le site de Xavié dupré 
from pymyinstall.fix import fix_scipy10_for_statsmodels08
fix_scipy10_for_statsmodels08()


# In[94]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[95]:


df = df.reset_index(drop=True)
df.tail()


# In[48]:



cols = ['symbol', 'date', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen',
       'adjVolume', 'close', 'divCash', 'high', 'low', 'open', 'splitFactor',
       'volume']
mycols = ['close',  'high', 'low', 'open', 'volume']

df2 = df.reset_index(drop=True)[mycols]
print(df2.shape)
df2.head()
data = df2['close']


# In[203]:


df_pandas = df2.copy()
df_pandas["yesterday_close"] = data.shift(1)
df_pandas["delta"] = df_pandas["close"] - df_pandas["yesterday_close"]
df_pandas.head()


# In[204]:


index_with_nan = [0]
df_pandas.drop(index_with_nan, 0, inplace = True)


# In[206]:


df_pandas.head()


# In[207]:


df_output = df_pandas['delta']


# In[208]:


array_output = df_output.to_numpy()


# In[209]:


array_output.shape


# ## Stat desc

# In[295]:


plt.gcf().set_size_inches(15, 10)
plt.plot(array_output)
plt.show()


# In[211]:


df_pandas.boxplot(column = 'delta')


# In[212]:


df_pandas.head()


# In[213]:


import numpy as np
import statsmodels.api as sm
import pylab


sm.qqplot( (df_pandas.delta- df_pandas.delta.mean())/df_pandas.delta.std(), line='45')
plt.gcf().set_size_inches(15, 10)
pylab.show()


# In[214]:


((df_pandas.delta - df_pandas.delta.mean())/df_pandas.delta.std()).kurt()


# In[215]:


# from scipy import stats

# dt =(df_pandas.delta- df_pandas.delta.mean())/df_pandas.delta.std()
# dt[(np.abs(stats.zscore(dt)) < 3)]


# In[304]:


scaler = MinMaxScaler(feature_range = (0,1))
df_pandas_train = df_pandas[["close","high", "low", "open", "volume"]] #on n'utilise pas la colonne "yesterday_close"
array_pandas_standardise = scaler.fit_transform(df_pandas_train)


# In[217]:


print(array_pandas_standardise[:10].reshape(-1,5).shape)


# In[218]:


df_pandas[0:1] 
#df_pandas[1:2]


# In[352]:


n = len(array_output)
n


# In[353]:


def split_data(array_train, array_output, prop_train_size, nb_columns):
    """
    
    """
    
    n = len(array_output)
    training_size = int(n*prop_train_size)
    test_size = n- training_size
    
    array_split_train = array_train[:training_size].reshape(-1,nb_columns)
    array_split_test = array_train[training_size:].reshape(-1,nb_columns)
    
    array_split_train_output = array_output[:training_size].reshape(-1,1)
    array_split_test_output = array_output[training_size:].reshape(-1,1)

    
    return array_split_train, array_split_test, array_split_train_output, array_split_test_output

def create_windows_data_XandY(array_split_train, array_split_train_output, time_step = 15):
    dataX = []
    dataY = []
    for i in range(len(array_split_train_output)- time_step - 1) : 
        set_value = array_split_train[i : (i+time_step) ,:]
        dataX.append(set_value)
        dataY.append(array_split_train_output[i + time_step])
    return np.array(dataX), np.array(dataY)


# In[306]:


prop_train_size = 0.65
nb_columns = 5
train_data, test_data ,train_data_y, test_data_y = split_data(array_pandas_standardise, array_output, prop_train_size, nb_columns)


# In[307]:


train_data[1:10,:]


# In[308]:


print(train_data.shape)
print(test_data.shape)
print(type(train_data))
print(type(test_data))


# In[309]:


print(train_data_y.shape)
print(test_data_y.shape)
print(type(train_data_y))
print(type(test_data_y))


# In[329]:


time_step = 200
X_train, Y_train = create_windows_data_XandY(train_data, train_data_y, time_step = 100)
X_test, Y_test = create_windows_data_XandY(test_data, test_data_y, time_step = 100)


# In[330]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[331]:


model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape= (time_step, nb_columns))) #les input shape doivent etre les memes que les deux valeurs de fin de la fonction reshape  (X_train.shape[1], 1) definis dans la cellule un peu avant !
model.add(Dropout(0.3))
model.add(LSTM(50, return_sequences = True, input_shape= (time_step, nb_columns)))
model.add(Dropout(0.3))
model.add(LSTM(50)) 
model.add(Dense(1))


# In[332]:


model.compile(loss =  'mean_squared_error', optimizer = 'adam')


# In[333]:


model.summary()


# In[334]:


model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 200, batch_size = 64, verbose=1)


# In[335]:


import tensorflow as tf
tf.__version__


# In[336]:


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[337]:


print(train_predict.shape)
print(test_predict.shape)


# In[338]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(Y_train,train_predict))


# In[339]:


math.sqrt(mean_squared_error(Y_test,test_predict))


# In[365]:


#plt.plot(array_output[200:817])
plt.plot(Y_train)
plt.plot(train_predict)
plt.gcf().set_size_inches(15, 10)

plt.show()


# In[363]:


#plt.plot(array_output[917:])
plt.plot(Y_test)

plt.plot(test_predict)
plt.gcf().set_size_inches(15, 10)

plt.show()


# In[354]:


training_size = int(n*prop_train_size)
test_size = n- training_size


# In[355]:


training_size


# In[357]:


len(train_predict)


# In[362]:


len(Y_train)


# In[ ]:




