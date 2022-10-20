#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("Alloy_Features_all_data.csv")
data = data.iloc[:,1:data.shape[1]]
features = data.iloc[:,1:data.shape[1]-1]


# In[3]:


features


# In[4]:


correlation = features.corr()
correlation


# In[5]:


cor = np.array(correlation)
cor


# In[8]:


labels = [r'$\Delta a$',r'$\Delta T_{m}$',r'$\Delta H_{mix}$',r'$a_{m}$',r'$\lambda$',r'$\delta$',r'$\Omega$',r'$T_{m}$',r'$\Delta \chi_{Allen}$',r'$\Delta \chi_{Pauling}$',r'$\Delta G$',r'$G$',r'$\Delta S_{mix}$','V.E.C']


# In[10]:


f, ax = plt.subplots(figsize = (16,16))
sns.heatmap(cor,annot = True)
ax.set_xticklabels(labels,fontsize = 15)
ax.set_yticklabels(labels,fontsize = 15)
plt.savefig('corr_refractory.png')
plt.show


# In[ ]:




