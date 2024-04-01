#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


from sklearn.datasets import load_boston

load_boston = load_boston()

X = load_boston.data
Y = load_boston.target

data = pd.DataFrame(X, columns=load_boston.feature_names)
data["SalesPrice"] = Y #salesprice
data.head()






# In[20]:


print(load_boston.DESCR)


# In[21]:


print(data.shape)


# In[22]:


data.info()


# In[23]:


data.describe()


# In[25]:


data.isnull().sum()


# In[26]:


sns.pairplot(data, height = 2.5)
plt.tight_layout()


# In[27]:


sns.distplot(data['SalesPrice']);


# In[28]:


print("Skewness: %f" %data['SalesPrice'].skew())
print("Kurtosis: %f" %data['SalesPrice'].kurt())


# In[29]:


fig, ax = plt.subplots()
ax.scatter(x = data['AGE'], y =data['SalesPrice'])
plt.ylabel('SalesPrice', fontsize = 13)
plt.xlabel('CRIM', fontsize = 13)
plt.show()


# In[34]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics

sns.distplot(data['SalesPrice'], fit=norm);

(mu, sigma) = norm.fit(data['SalesPrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n '.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],loc = 'best')
plt.ylabel('Frequency')
plt.title('SalesPrice distribution')

fig = plt.figure()
res = stats.probplot(data['SalesPrice'], plot=plt)
plt.show()


# In[40]:


data["SalesPrice"] = np.log1p(data["SalesPrice"])

sns.distplot(data['SalesPrice'] , fit=norm);
(mu, sigma) = norm.fit(data['SalesPrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(data['SalesPrice'], plot=plt)
plt.show()


# DAta COrrelation
# 

# In[41]:


plt.figure(figsize=(10,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.PuBu)
plt.show()


# In[43]:


cor_target = abs(cor["SalesPrice"]) # absolute value of the correlation 

relevant_features = cor_target[cor_target>0.2] # highly correlated features 

names = [index for index, value in relevant_features.iteritems()] # getting the names of the features 

names.remove('SalesPrice') # removing target feature 

print(names) # printing the features 
print(len(names))


# Model Building

# In[45]:


from sklearn.model_selection import train_test_split 

X = data.drop("SalesPrice", axis=1) 
y = data["SalesPrice"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[46]:



print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[52]:


from sklearn.linear_model import LinearRegression 

lr = LinearRegression() 
lr.fit(X_train, y_train)


# In[53]:


predictions = lr.predict(X_test)  

print("Actual value of the house:- ", y_test[0]) 
print("Model Predicted Value:- ", predictions[0])


# In[49]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions) 
rmse = np.sqrt(mse)
print(rmse)


# In[ ]:




