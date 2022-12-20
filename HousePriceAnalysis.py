# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 22:08:09 2022

@author: ugura
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']=400
from sklearn.linear_model import LinearRegression

pd.options.display.max_columns = None

df=pd.read_csv('C:/WorkSamples/HousePricePrediction/realtor-data.csv')
df.head()
df.info()

df=df.drop_duplicates()

df.info()

df=df.dropna()
df.info()

full_address_counts = df['full_address'].value_counts()
full_address_counts.head()

df=df.drop_duplicates(subset=['full_address'],keep= 'last')
full_address_counts = df['full_address'].value_counts()
full_address_counts.head()
df.info()

df.to_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean.csv',index=False)
df=pd.read_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean.csv')

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

df['bath'].describe()
df.groupby('bath').agg({'price':'mean'}).plot.bar(legend=False)
plt.ylabel('Avg Price')
plt.xlabel('Total Number of Bathroom')

bath_counts = df['bath'].value_counts()
bath_counts
df[hist_columns[2]].hist()
df.drop(df[(df['bath'] >10)].index, inplace=True)
df.info()

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

df.to_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean_Without_Outliers.csv',index=False)














