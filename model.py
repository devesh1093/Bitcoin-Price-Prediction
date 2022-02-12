#!/usr/bin/env python
# coding: utf-8

# ## Bitcoin Price Prediction using Prophet() Model

# ### Importing all the necessary Libraries

# In[1]:


import requests
import pandas as pd 
import numpy as np
from fbprophet import Prophet
from matplotlib import pyplot
import datetime
from sklearn.metrics import mean_absolute_error,r2_score, max_error,explained_variance_score,mean_squared_error
from fbprophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics


# In[ ]:





# ### Making request to API of coingecko.com for Bitcoin historical Data

# In[2]:


#Enter the amount of days in past upto which the data is to be requested
a=input("Enter the days:")
#b=input("Enter the currency:")
res=requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=INR&days="+a+"&interval=daily")
data=res.json()


# In[3]:


data


# ### Constructing the dataset

# In[4]:


today = datetime.datetime.today()

dateList = []
for x in range (0, int(a)+1):
    dateList.append((today - datetime.timedelta(days = x)).strftime('%Y-%m-%d'))
dateList=dateList[::-1]
df = pd.DataFrame({ 'Date': dateList}) 
df


# In[5]:


file=data['prices']
lis=[]
for i in file:
    lis.append(i[1])
lis
data=[lis]
df["Price"]=pd.DataFrame({'Price':lis})

df['Date']=pd.to_datetime(df['Date'])


# In[6]:


df


# In[7]:


df.info()


# In[8]:


df.describe().transpose()


# 

# In[9]:


#Following Plot shows the trends in the given dataset.
df.plot(x='Date',y='Price',kind='line')
pyplot.show()


# In[10]:


df.corr()


# We need to change the name of labels to 'ds' and 'y' to implement Prophet

# In[11]:


df.rename(columns={"Date":"ds","Price":"y"},inplace=True)
df


# ###  Splitting into Training and Testing data

# In[12]:


# Train test split
df_train = df.iloc[0:int(0.8*len(df))]
df_test = df.iloc[int(0.8*len(df)):]
# Print the number of records and date range for training and testing dataset.
print('The training dataset has', len(df_train), 'records, ranging from', df_train['ds'].min(), 'to', df_train['ds'].max())
print('The testing dataset has', len(df_test), 'records, ranging from', df_test['ds'].min(), 'to', df_test['ds'].max())


# In[13]:


df_train


# In[14]:


df_test


# In[ ]:





# ### Implementing the prophet model

# In[15]:


# Create the prophet model with confidence internal of 95%
m = Prophet(interval_width=0.95, n_changepoints=20)
# Fit the model using the dataset
m.fit(df)


# In[ ]:





# ###  In-Sample Forecast on Testing data

# In[16]:


in_sample=df_test
# use the model to make a forecast
forecast_in = m.predict(in_sample)
# summarize the forecast
print(forecast_in[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
# plot forecast
m.plot(forecast_in)
pyplot.show()


# ### Model Performance 

# In[17]:


y_true = df_test['y'].values
y_pred = forecast_in['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
r2=r2_score(y_true,y_pred)
max_err=max_error(y_true,y_pred)
mse = mean_squared_error(y_true,y_pred)
evs=explained_variance_score(y_true,y_pred)
print('MAE: %.3f' % mae)
print('R2 SCORE: %.3f' % r2)
print('MAX ERROR: %.3f' % max_err)
print('MSE: %.3f' % mse)
print('EVS: %.3f' % evs)

# plot expected vs actual
pyplot.plot(y_true, label='Actual')
pyplot.plot(y_pred, label='Predicted')
pyplot.legend()
pyplot.show()


# In[ ]:





# ### Out Sample Forecast

# In[18]:


# Create a future dataframe for prediction
pred_days=int(input())
future = m.make_future_dataframe(periods=pred_days)
future.tail(pred_days)


# In[19]:



# Forecast the future dataframe values
forecast_out = m.predict(future)
# Check the forecasted values and upper/lower bound
forecast_out[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[20]:


# Visualize the forecast
fig = m.plot(forecast_out)
ax = fig.gca()
ax.plot(df_test["ds"], y_pred, 'r')

pyplot.show()


# In[21]:


m.plot_components(forecast_out);


# In[22]:


# Change points to plot
fig = m.plot(forecast_out)
a = add_changepoints_to_plot(fig.gca(), m, forecast_out)


# In[ ]:





# In[ ]:




