#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#Exploratory Data Analysis


# In[3]:


data = pd.read_csv("customer_credit_data.csv")


# In[4]:


data.head()


# In[5]:


data = data.drop("Unnamed: 0", axis=1)


# In[6]:


data.shape


# In[7]:


data = data.rename(columns={'V1': 'chk_acct', 'V2' : "duration", 'V3' : "credit_his", 'V4' : "purpose", 'V5' : "amount", 'V6' : "saving_acct", 'V7' : "present_emp", 'V8' : "installment_rate", 'V9' : "sex", 'V10' : "other_debtor", 'V11' : "present_resid", 'V12' : "property", 'V13' : "age", 'V14' : "other_install", 'V15' : "housing", 'V16' : "n_credit", 'V17' : "job", 'V18' : "n_people", 'V19' : "telephone", 'V20' : "foreign", 'V21' : "response"})


# In[8]:


data.head()


# In[9]:


data["response"] = data["response"] - 1


# In[10]:


data["response"].head()


# In[11]:


data.dtypes


# In[12]:


data.describe()


# In[13]:


data.corr()


# In[14]:


data.hist(column="installment_rate", by="response")


# In[15]:


data.boxplot(column="age", by="response")


# In[16]:


data.boxplot(column="duration", by="response")


# In[17]:


#Comment


# In[18]:


data.hist(column="chk_acct", by="response")


# In[19]:


data.hist(column="credit_his", by="response")


# In[20]:


data.hist(column="saving_acct", by="response")


# In[21]:


#Saving the cleaned data to csv


# In[22]:


data.to_csv("cleaned_credit_data.csv")


# In[ ]:




