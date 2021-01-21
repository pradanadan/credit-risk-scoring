#!/usr/bin/env python
# coding: utf-8

# In[25]:


get_ipython().system(' pip install sklearn')


# In[1]:


import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from numpy import save


# In[2]:


data = pd.read_csv("cleaned_credit_data.csv")


# In[3]:


data.head()


# In[4]:


predict = "response"

X = data[["sex", "housing", "saving_acct", "chk_acct", "age", "duration", "amount"]]
y = np.array(data[predict])


# In[5]:


X.head()


# In[6]:


#ONE HOT ENCODING THE CATEGORICAL DATA AND NORMALIZE THE NUMERICAL DATA


# In[7]:


#One hot encoding
ohe = make_column_transformer((OneHotEncoder(), ["sex", "housing", "saving_acct", "chk_acct"]), remainder="passthrough")
x = ohe.fit_transform(X)

#Normalize to 0 to 1
normalize = MinMaxScaler()
normalize.fit(x)
x = normalize.transform(x)


# In[8]:


print(x)


# In[9]:


#DIVIDING DATA INTO TRAIN, VALIDATION, AND TES 0.8-0.1-0.1


# In[10]:


x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.1)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_train, test_size=(0.1/0.9))


# In[11]:


#LOGISTIC REGRESSION


# In[12]:


logisticReg = linear_model.LogisticRegression().fit(x_train, y_train)
logisticReg.score(x_test, y_test)


# In[13]:


y_predicted = logisticReg.predict(x_test)
confusion_matrix(y_test, y_predicted)


# In[14]:


#The number of true positive is 64; false positive is 5; false negative is 18; and true negative is 13 


# In[15]:


#Logistic Regression, SVC, Random Forest


# In[16]:


kf = KFold(n_splits=10)

acc_logisticReg = 0
acc_svc = 0
acc_ranForest = 0

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    logisticReg = linear_model.LogisticRegression()
    logisticReg.fit(x_train, y_train)
    acc1 = logisticReg.score(x_test, y_test)

    svc = svm.SVC()
    svc.fit(x_train, y_train)
    acc2 = svc.score(x_test, y_test)
    
    ranForest = RandomForestClassifier(max_depth=20)
    ranForest.fit(x_train, y_train)
    acc3 = ranForest.score(x_test, y_test)
    
    if acc1 > acc_logisticReg:
        acc_logisticReg = acc1
        with open("logisticReg.pickle", "wb") as f:
            pickle.dump(logisticReg, f)
    
    if acc2 > acc_svc:
        acc_svc = acc2
        with open("SVC.pickle", "wb") as f:
            pickle.dump(svc, f)
            
    if acc3 > acc_ranForest:
        acc_ranForest = acc3
        with open("ranForest.pickle", "wb") as f:
            pickle.dump(ranForest, f)


# In[18]:


print("Accuracy of Logistic Regression: ", acc_logisticReg)
print("Accuracy of SVC: ", acc_svc)
print("Accuracy of Random Forest: ", acc_ranForest)


# In[19]:


#Save numpy data to a file
save("x_data.npy", x)
save("y_data.npy", y)


# In[20]:


#Load models from pickle


# In[21]:


pickle_in = open("logisticReg.pickle", "rb")
logisticReg = pickle.load(pickle_in)

pickle_in = open("SVC.pickle", "rb")
svc = pickle.load(pickle_in)

pickle_in = open("ranForest.pickle", "rb")
ranForest = pickle.load(pickle_in)


# In[ ]:




