# RealEstatePriceAnalysisWithPython
Dataset : https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset <br>

import pandas as pd <br>
import numpy as np <br>
import matplotlib.pyplot as plt <br>
import matplotlib as mpl <br>
mpl.rcParams['figure.dpi']=400 <br> 
from sklearn.linear_model import LinearRegression <br>

pd.options.display.max_columns = None <br>

df=pd.read_csv('C:/WorkSamples/HousePricePrediction/realtor-data.csv') <br>
df.head() <br>
df.info() <br>

df=df.drop_duplicates() <br>

df.info() <br>

df=df.dropna() <br>
df.info() <br>

full_address_counts = df['full_address'].value_counts() <br>
full_address_counts.head()

df=df.drop_duplicates(subset=['full_address'],keep= 'last') <br>
full_address_counts = df['full_address'].value_counts() <br>
full_address_counts.head() <br>
df.info() <br>

df.to_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean.csv',index=False) <br>
df=pd.read_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean.csv') <br>

df['bed'].describe() <br>
df.groupby('bed').agg({'price':'mean'}).plot.bar(legend=False) <br>
plt.ylabel('Avg Price') <br>
plt.xlabel('Total Number of Bedroom') <br>

bed_counts = df['bed'].value_counts() <br>
bed_counts <br>

hist_columns=['price','bed','bath','house_size','acre_lot'] <br>
df[hist_columns[1]].hist() <br>
 
df.drop(df[(df['bed'] >10) | (df['bed'] < 1)].index, inplace=True) <br>
df.info() <br>
 
df['bath'].describe() <br>
df.groupby('bath').agg({'price':'mean'}).plot.bar(legend=False) <br>
plt.ylabel('Avg Price') <br>
plt.xlabel('Total Number of Bathroom') <br>

bath_counts = df['bath'].value_counts() <br>
bath_counts <br>
df[hist_columns[2]].hist() <br>
df.drop(df[(df['bath'] >10)].index, inplace=True) <br>
df.info() <br>

df['house_size'].describe() <br>
df['house_size_group']=np.where(df['house_size']<10000, '1-<10K', <br>
                       np.where(df['house_size']<20000, '2-10K-20K', <br>
                       np.where(df['house_size']<30000, '3-20K-30K', <br>
                       np.where(df['house_size']<40000, '4-30K-40K', <br>
                       '5->40K')))) <br>
df.groupby('house_size_group').agg({'price':'mean'}).plot.bar(legend=False) <br>
plt.ylabel('Avg Price') <br>
plt.xlabel('Total Number of house_size_group') <br>

house_size_group_counts = df['house_size_group'].value_counts() <br>
house_size_group_counts <br> 
df[hist_columns[3]].hist() <br>

![TotalNumberOfHouseSizeGroup](https://user-images.githubusercontent.com/114496063/208690428-c73d6418-23ef-43f7-a434-a9f53770e88d.png)

df.drop(df[(df['house_size'] >10000)].index, inplace=True) <br>
df.info() <br>

df['acre_lot'].describe() <br>
acre_lot_counts = df['acre_lot'].value_counts() <br>
acre_lot_counts <br>

np.percentile(df['acre_lot'], 90) <br>
np.percentile(df['acre_lot'], 91) <br>
np.percentile(df['acre_lot'], 92) <br>
np.percentile(df['acre_lot'], 93) <br>
np.percentile(df['acre_lot'], 94) <br>
np.percentile(df['acre_lot'], 95) <br>
np.percentile(df['acre_lot'], 96) <br>
np.percentile(df['acre_lot'], 97) <br>
np.percentile(df['acre_lot'], 98) <br>
np.percentile(df['acre_lot'], 99) <br>


df.drop(df[(df['acre_lot'] >7.59) |(df['acre_lot'] ==0)].index, inplace=True) <br>
df.info() <br>

df['price(M)']=df['price']/1000000 <br>
df['price(M)'].describe() <br>


df['price_group']=np.where(df['price(M)']<0.275, '1-<0.275M', <br>
                       np.where(df['price(M)']<0.425, '2-0.275M-0.425M', <br>
                       np.where(df['price(M)']<0.699, '3-0.425M-0.699M', <br>
                       np.where(df['price(M)']<1, '4-0.699M-1M', <br>
                       np.where(df['price(M)']<1.5, '5-1M-1.5M',      
                       '6->1.5M'))))) <br>

price_group_counts = df['price_group'].value_counts() <br>
price_group_counts <br>

np.percentile(df['price(M)'], 90) <br>
np.percentile(df['price(M)'], 91) <br>
np.percentile(df['price(M)'], 92) <br>
np.percentile(df['price(M)'], 93) <br>
np.percentile(df['price(M)'], 94) <br>
np.percentile(df['price(M)'], 95) <br>
np.percentile(df['price(M)'], 96) <br>
np.percentile(df['price(M)'], 97) <br>
np.percentile(df['price(M)'], 98) <br>
np.percentile(df['price(M)'], 99) <br>


df.drop(df[(df['price(M)'] >3.295)].index, inplace=True) <br>
df.info() <br>

df.to_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean_Without_Outliers.csv',index=False) <br>
