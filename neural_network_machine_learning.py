#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system(' pip install tensorflow')


# In[61]:


import tensorflow
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold


# In[2]:


#Opening Numpy Array Data
#Those array are credit_credit_data.csv those have been processed in "sklearn_machine_learning.py"


# In[62]:


x = np.load("x_data.npy")
y = np.load("y_data.npy")


# In[63]:


#Making keras model


# In[64]:


model = keras.Sequential()
model.add(layers.InputLayer(19))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))


# In[65]:


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])


# In[66]:


#Run the model with cross validation


# In[68]:


kf = KFold(n_splits=10)
best_acc = 0

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(x_train, y_train, epochs=100, batch_size=80)
    acc = model.evaluate(x_test, y_test)
    
    if acc[1] > best_acc:
        best_acc = acc[1]
        model.save("neuralNetModel.h5")


# In[69]:


best_acc


# In[ ]:




