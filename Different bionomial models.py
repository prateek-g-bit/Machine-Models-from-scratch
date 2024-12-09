#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer    #imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
import statsmodels.api as sm


# In[2]:


wine = datasets.load_wine()
print(wine.DESCR)


# In[3]:


dir(wine)


# In[4]:


wine.feature_names


# In[5]:


df = pd.DataFrame(wine.data,columns=wine.feature_names)


# In[6]:


df.head(10)


# In[7]:


wine.target


# In[8]:


wine.target_names


# In[9]:


X_train, X_test, y_train, y_test =train_test_split(wine.data,wine.target, test_size=0.2, random_state=42)


# In[10]:


from sklearn.naive_bayes import MultinomialNB,GaussianNB
mnb=MultinomialNB()
mnb.fit(X_train,y_train)


# In[11]:


mnb.score(X_train,y_train)


# In[12]:


gnb=GaussianNB()
gnb.fit(X_train,y_train)


# In[13]:


gnb.score(X_train,y_train)


# In[14]:


mnb.fit(X_test,y_test)


# In[15]:


mnb.score(X_test,y_test)


# In[16]:


gnb.fit(X_test,y_test)
gnb.score(X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:




