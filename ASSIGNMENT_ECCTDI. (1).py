#!/usr/bin/env python
# coding: utf-8

# In[3]:


###############################################################################################

########3 USING ECCTDI INDEX AND ANALYSING WITH GIVEN DATASET #############################################



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


data_excel = pd.read_excel(r"C:/Users/shiva/Downloads/rainfall_monthly.xlsx")


# In[4]:


print(data_excel)


# In[5]:


from scipy import stats


# In[32]:


df = pd.read_excel(r"C:/Users/shiva/Downloads/daily_rainfall.xlsx")


# In[33]:


print(df)


# In[34]:


df.index = pd.date_range('01-01-1901',periods = df.shape[0], freq = 'D')


# In[35]:


df1 = pd.DataFrame(df)


# In[31]:


print(df1)


# In[37]:


df1


# In[38]:


df1.columns = ['l1','l2','l3','l4','l5','l6','l7']


# In[39]:


df1


# In[63]:


df1 = df1[df1['l2'] >= 25]


# In[65]:


df1['year'] = df1.index.year


# In[66]:


df3 = df1.groupby('year')['l2'].count()


# In[59]:


for col in df3.columns:
    print(col)


# In[67]:


df3.plot(x="year", y="values", kind="line")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




