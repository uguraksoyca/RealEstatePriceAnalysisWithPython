#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']=400
from sklearn.linear_model import LinearRegression
pd.options.display.max_columns = None


# In[ ]:


df=pd.read_csv('C:/WorkSamples/HousePricePrediction/realtor-data.csv')
df.head()
df.info()


# In[ ]:


df=df.drop_duplicates()
df.info()


# In[ ]:


df=df.dropna()
df.info()


# In[ ]:


full_address_counts = df['full_address'].value_counts()
full_address_counts.head()


# In[ ]:


df=df.drop_duplicates(subset=['full_address'],keep= 'last')
full_address_counts = df['full_address'].value_counts()
full_address_counts.head()
df.info()


# In[ ]:


df.to_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean.csv',index=False)
df=pd.read_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean.csv')


# In[ ]:


df['bed'].describe()
df.groupby('bed').agg({'price':'mean'}).plot.bar(legend=False)
plt.ylabel('Avg Price')
plt.xlabel('Total Number of Bedroom')

bed_counts = df['bed'].value_counts()
bed_counts

hist_columns=['price','bed','bath','house_size','acre_lot']
df[hist_columns[1]].hist()

df.drop(df[(df['bed'] >10) | (df['bed'] < 1)].index, inplace=True)
df.info()


# In[ ]:


df['bath'].describe()
df.groupby('bath').agg({'price':'mean'}).plot.bar(legend=False)
plt.ylabel('Avg Price')
plt.xlabel('Total Number of Bathroom')

bath_counts = df['bath'].value_counts()
bath_counts
df[hist_columns[2]].hist()
df.drop(df[(df['bath'] >10)].index, inplace=True)
df.info()


# In[ ]:


df['house_size'].describe()
df['house_size_group']=np.where(df['house_size']<10000, '1-<10K',
                       np.where(df['house_size']<20000, '2-10K-20K',
                       np.where(df['house_size']<30000, '3-20K-30K',
                       np.where(df['house_size']<40000, '4-30K-40K',
                       '5->40K'))))
df.groupby('house_size_group').agg({'price':'mean'}).plot.bar(legend=False)
plt.ylabel('Avg Price')
plt.xlabel('Total Number of house_size_group')

house_size_group_counts = df['house_size_group'].value_counts()
house_size_group_counts
df[hist_columns[3]].hist()

df.drop(df[(df['house_size'] >10000)].index, inplace=True)
df.info()


# In[ ]:


df['acre_lot'].describe()
acre_lot_counts = df['acre_lot'].value_counts()
acre_lot_counts

np.percentile(df['acre_lot'], 90)
np.percentile(df['acre_lot'], 91)
np.percentile(df['acre_lot'], 92)
np.percentile(df['acre_lot'], 93)
np.percentile(df['acre_lot'], 94)
np.percentile(df['acre_lot'], 95)
np.percentile(df['acre_lot'], 96)
np.percentile(df['acre_lot'], 97)
np.percentile(df['acre_lot'], 98)
np.percentile(df['acre_lot'], 99)


df.drop(df[(df['acre_lot'] >7.59) |(df['acre_lot'] ==0)].index, inplace=True)
df.info()


# In[ ]:


df['price(M)']=df['price']/1000000
df['price(M)'].describe()


df['price_group']=np.where(df['price(M)']<0.275, '1-<0.275M',
                       np.where(df['price(M)']<0.425, '2-0.275M-0.425M',
                       np.where(df['price(M)']<0.699, '3-0.425M-0.699M',
                       np.where(df['price(M)']<1, '4-0.699M-1M',
                       np.where(df['price(M)']<1.5, '5-1M-1.5M',       
                       '6->1.5M')))))

price_group_counts = df['price_group'].value_counts()
price_group_counts

np.percentile(df['price(M)'], 90)
np.percentile(df['price(M)'], 91)
np.percentile(df['price(M)'], 92)
np.percentile(df['price(M)'], 93)
np.percentile(df['price(M)'], 94)
np.percentile(df['price(M)'], 95)
np.percentile(df['price(M)'], 96)
np.percentile(df['price(M)'], 97)
np.percentile(df['price(M)'], 98)
np.percentile(df['price(M)'], 99)


df.drop(df[(df['price(M)'] >3.295)].index, inplace=True)
df.info()


# In[ ]:


df.to_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean_Without_Outliers.csv',index=False)


# In[3]:


df=pd.read_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean_Without_Outliers.csv')


# In[4]:


corrM = df.corr()
 
"""
We can see that bath and house_size has more that 0.6 correlation with price.
We can use these variables to create linear regression models.
"""
corrM


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# # Single Linear Regression

# In[61]:


X_train,X_Test,y_train,y_test=train_test_split(df['bath'].values.reshape(-1,1),
                                              df['price'].values,test_size=0.2,random_state=24
                                        )


# In[62]:


print(X_train.shape)
print(X_Test.shape)
print(y_train.shape)
print(y_test.shape)


# In[63]:


model = LinearRegression()
model


# In[64]:


model.fit(X_train, y_train)


# In[65]:


model_pred=model.predict(X_Test)


# In[66]:


mae = mean_absolute_error(y_test, model_pred)
r2 = r2_score(y_test, model_pred)
r2_percent = r2*100


# In[70]:


print(f"The accuracy of price based on bath predictor is: {r2_percent:.2f}%")


# In[71]:


print(f"On average, this listing price based on bath predictor is $ {mae:.2f} off of any given price.")


# In[69]:


X_train2,X_Test2,y_train2,y_test2=train_test_split(df['house_size'].values.reshape(-1,1),
                                              df['price'].values,test_size=0.2,random_state=24
                                        )


# In[78]:


model2 = LinearRegression()
model2.fit(X_train2, y_train2)
model_pred2=model2.predict(X_Test2)
mae_2 = mean_absolute_error(y_test2, model_pred2)
r2_2 = r2_score(y_test2, model_pred2)
r2_percent_2 = r2_2*100


# In[79]:


print(f"The accuracy of price based on house_size predictor is: {r2_percent_2:.2f}%")


# In[80]:


print(f"On average, this listing price based on house_size predictor is $ {mae_2:.2f} off of any given price.")


# ## Multi Linear Regression

# In[82]:


x3 = df[['bed', 'bath', 'acre_lot','house_size']]
y3 = df['price']


# In[83]:


x_train3, x_test3,y_train3, y_test3 = train_test_split(x3, y3, test_size = 0.2, random_state = 100)


# In[84]:


model3 = LinearRegression()  
model3.fit(x_train3, y_train3)


# In[86]:


model_pred3= model3.predict(x_test3)


# In[91]:


model_pred3_diff = pd.DataFrame({'Actual value': y_test3, 'Predicted value': model_pred3})
model_pred3_diff.head()


# In[92]:


mae_3 = mean_absolute_error(y_test3, model_pred3)
r2_3 = r2_score(y_test3, model_pred3)
r2_percent_3 = r2_3*100


# In[93]:


print(f"The accuracy of price based on house_size predictor is: {r2_percent_3:.2f}%")


# In[94]:


print(f"On average, this listing price based on house_size predictor is $ {mae_3:.2f} off of any given price.")


# # Random Forest Regression

# In[126]:


x4 = df.bath.values.reshape(-1, 1)
y4 = df.price.values


# In[127]:


x_train4, x_test4,y_train4, y_test4 = train_test_split(x4, y4, test_size = 0.2, random_state = 40)


# In[128]:


model4 = RandomForestRegressor()
model4.fit(x_train4, y_train4)


# In[129]:


# Test Model
model_pred4 = model4.predict(x_test4)

# Get Test Scores
mae_4 = mean_absolute_error(y_test4, model_pred4)
r2_4 = r2_score(y_test4, model_pred4)
r2_percent_4 = r2_4*100


# In[130]:


print(f"The accuracy of price based on bath predictor is: {r2_percent_4:.2f}%")


# In[131]:


print(f"On average, this listing price based on bath predictor is $ {mae_4:.2f} off of any given price.")


# In[132]:


x5 = df.house_size.values.reshape(-1, 1)
y5 = df.price.values
x_train5, x_test5,y_train5, y_test5 = train_test_split(x5, y5, test_size = 0.2, random_state = 40)
model5 = RandomForestRegressor()
model5.fit(x_train5, y_train5)

# Test Model
model_pred5 = model5.predict(x_test5)

# Get Test Scores
mae_5 = mean_absolute_error(y_test5, model_pred5)
r2_5 = r2_score(y_test5, model_pred5)
r2_percent_5 = r2_5*100


# In[133]:


print(f"The accuracy of price based on house_size predictor is: {r2_percent_5:.2f}%")


# In[134]:


print(f"On average, this listing price based on bath predictor is $ {mae_5:.2f} off of any given price.")


# In[135]:


X6 = df.drop("price", axis = 1)
y6 = df["price"]


# In[137]:


y6


# In[138]:


X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, test_size = 0.2)


# In[142]:


model6 = RandomForestRegressor()
model6.fit(X_train6.values, y_train6)


# In[143]:


model_pred6 = model6.predict(X_test6.values)


# In[147]:


mae_6 = mean_absolute_error(y_test6, model_pred6)
r2_6 = r2_score(y_test6, model_pred6)
r2_percent_6 = r2_6*100


# In[149]:


model_pred6_diff = pd.DataFrame({'Actual value': y_test6, 'Predicted value': model_pred6})
model_pred6_diff.head()


# In[145]:


print(f"The accuracy of price based on house_size predictor is: {r2_percent_6:.2f}%")


# In[148]:


print(f"On average, this listing price based on bath predictor is $ {mae_6:.2f} off of any given price.")


# In[153]:


results = pd.DataFrame({'SL_Bath_and_Price': [mae, r2, r2_percent],
                   'SL_House_Size_and_Price': [mae_2, r2_2, r2_percent_2],
                   'ML': [mae_3, r2_3, r2_percent_3],
                   'RF_Bath_and_Price': [mae_4, r2_4, r2_percent_4],
                   'RF_House_Size_and_Price': [mae_5, r2_5, r2_percent_5],
                   'RF_All_Columns_For_Price': [mae_6, r2_6, r2_percent_6]},
                  index=['mean_absolute_error', 'r2_score', 'r2_percent'])


# In[156]:


results


# In[ ]:




