#!/usr/bin/env python
# coding: utf-8

# In[1]:


#################CLASS1######################3
#####################PLOTTING GRAPHS FOR LATITUDINAL CHANGE IN PRECIPITATION###########################
import xarray as xr


# In[2]:


dset = xr.open_dataset("C:/Users/shiva/Downloads/adaptor.mars.internal-1677233580.318813-7673-3-a4160bbc-ed0d-49c0-9333-909daa219018.nc")


# In[3]:


print(dset)


# In[4]:


dset['tp'].plot()


# In[5]:


dset['tp'].plot(cmap = 'jet', vmax = 0.02)


# In[ ]:





# In[ ]:


########CLASS2####################


# In[ ]:


##############PLOTS OF MKENDALL TEST##################


# In[ ]:


###########################################################################
#testing for MKENDALL TEST and plotting


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


# In[3]:


data = pd.read_csv(r"C:/Users/shiva/Downloads/co2.csv",header=1)


# In[9]:


pip install pymannkendall 


# In[10]:


import pymannkendall as mk


# In[11]:


data.head()


# In[12]:


a = mk.original_test(data['average'])


# In[12]:


a


# In[13]:


plt.plot(data['average'],data['month'])


# In[ ]:





# In[15]:


####################################################
# trend plot of co2

plt.plot(data['year'],data['average'],'y')
plt.title("Trends in co2")
plt.xlabel("Year")
plt.ylabel("co2 ppm")
z = np.polyfit(data['year'], data['average'],)


p = np.poly1d(z)
plt.plot(data["average"], p(data["average"]))
plt.show()


# In[ ]:





# In[ ]:


######################## SEASONALITY ##################33


# In[4]:


data


# In[152]:


df1 = pd.DataFrame((data['year']).unique())
df1['average'] = ""


# In[153]:


import pandas as pd       

new_data = data.loc[(data['month'] == 7) | (data['month'] == 8)]

dff = pd.DataFrame(new_data.groupby(['year'], as_index = False).sum())
dff


# In[154]:


type(dff)


# In[155]:


d= pd.DataFrame(dff[['year', 'average']])
plt.plot(d['year'],d['average'])  ####seasonality graph of c02 yearwise


# In[ ]:





# In[ ]:


###########################
#plotting various sin functions


# In[16]:


import math
import numpy as np


# In[17]:


x = np.arange(1,1200,1)
y1 = np.sin(x)+ np.sin(2*x)+ np.sin(3*x)
plt.plot(x,y1)


# In[18]:


x = np.arange(1,1200,10)
y1 = np.sin(x)+ np.sin(2*x)+ np.sin(3*x)
plt.plot(x,y1)


# In[19]:


x = np.arange(1,400,1)
y = np.sin(x)
plt.plot(x,y)


# In[20]:


x = np.arange(400,800,1)
y = np.sin(2*x)
plt.plot(x,y)


# In[21]:


x = np.arange(800,1200,1)
y = np.sin(3*x)
plt.plot(x,y)


# In[22]:


x = np.arange(800,1200,10)
y = np.sin(3*x)
plt.plot(x,y)


# In[ ]:





# In[ ]:


#################LAT LONG DATA OF RAINFALL###################


# In[23]:



#14 march 2023
import pandas as pd
import numpy as nm
import os


# In[ ]:


############data of the lat long of my area


# In[24]:


data_excel = pd.read_excel(r"C:/Users/shiva/Downloads/rainfall_monthly.xlsx")


# In[25]:


print(data_excel)


# In[26]:


data_bpl = data_excel[(data_excel['Lat']== 77.5) & (data_excel['Long']==23.5)]


# In[27]:


print(data_bpl)


# In[ ]:





# In[ ]:





# In[ ]:


##############################  ARIMA MODEL OF DATAPOINTS #########################3 ASSIGNMENT QUESTION


# In[190]:


#27 march
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from   statsmodels.tsa.stattools import acf,pacf


# In[193]:


x = [2.89,7.39,23.88,10.59,5.91,1.53,3.48,56.54,26.19,6.35,38.09,0.01,3.03,41.57,44.73,21.39,15.87,1.22,21.75,.21]

df=pd.DataFrame(x,columns=['t'])

sm.tsa.acf(x)

df['t-1']=df['t'].shift(1)
df.drop([0],axis=0)
fig=plt.figure(figsize=(12,5))
ax1=fig.add_subplot(211)
fig=plot_acf(df['t-1'].dropna(),ax=ax1)
ax2=fig.add_subplot(212)
fig=plot_pacf(df['t-1'].dropna(),ax=ax2,lags=8)


# In[182]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df['t'],order=(1,1,1))
model_f= model.fit()
model_f.summary()


# In[183]:


forecast_ARIMA=model_f.predict(start=1,end=23)
forecast_ARIMA


# In[184]:


plt.plot(df['t'],'g',forecast_ARIMA,'r')
plt.show()


# In[ ]:





# In[ ]:


##### SECOND SET OF DATA POINTS ###########################


# In[ ]:


########################################
#####plotting ARIMA MODEL


# In[185]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from   statsmodels.tsa.stattools import acf,pacf


# In[186]:


data = [963.65,965.03,961.18,959.43,957.68,953.42,950.11,952.44,952.25,956.88,963.66,963.36,965.56,964.3,960.91,956.9,952.18,950.71,952.54,951.43,955.06,959.01,962.60]

df = pd.DataFrame(data , columns = ['values'])
print(df)


# In[187]:


data1 = data[1:]
data2 = data[2:]


# In[188]:


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from   statsmodels.tsa.stattools import acf,pacf

# Display the autocorrelation plot of your time series
fig = plot_acf(df['values'], lags = 10)
plt.show()


# In[189]:


fig = plot_pacf(df['values'], lags = 10)
plt.show()


# In[50]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df['values'],order = (1,0,1),)
model_fit = model.fit()
model_fit.summary()


# In[51]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df['values'],order = (0,1,0),)
model_fit = model.fit()
model_fit.summary()


# In[52]:


forecast = model_fit.predict(start= 1,end= 24)
print(forecast)
df['forecast']= forecast


# In[53]:


df['values'].plot()
df['forecast'].plot()


# In[ ]:





# In[ ]:



################################## PLOTTING SEASONALITY OF PRECIPITATION IN MY LAT LONG REGION ########################
#


# In[ ]:


#3/4/23
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[121]:


data_excel = pd.read_excel(r"C:/Users/shiva/Downloads/rainfall_monthly.xlsx")
data_bpl = data_excel[(data_excel['Lat']== 77.5) & (data_excel['Long']==23.5)]


# In[20]:





# In[128]:


df11 = (data_bpl.T )
df11


# In[135]:


d6 = df11.drop('Lat')


# In[139]:


d7 = d6.drop('Long')
d7


# In[142]:


plt.plot(d7.index, d7[2203])


# In[147]:


d7['year'] = pd.DatetimeIndex(d7.index).year
d7['month'] = pd.DatetimeIndex(d7.index).month
d7


# In[148]:


new_data = d7.loc[(d7['month'] == 7) | (d7['month'] == 8)]

dff1 = pd.DataFrame(new_data.groupby(['year'], as_index = False).sum())
dff1


# In[149]:


plt.plot(dff1['year'], dff1[2203])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




