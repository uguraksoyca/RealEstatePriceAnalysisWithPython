# RealEstatePriceAnalysisWithPython
Dataset : https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset <br>

import pandas as pd <br>
import numpy as np <br>
import matplotlib.pyplot as plt <br>
import matplotlib as mpl <br>
mpl.rcParams['figure.dpi']=400 <br> 
from sklearn.linear_model import LinearRegression <br>

pd.options.display.max_columns = None <br>

## Read file, Check the data
df=pd.read_csv('C:/WorkSamples/HousePricePrediction/realtor-data.csv') <br>
df.head() <br>
df.info() <br>

# 1-) DATA VALIDATION  <br>
## 1.1) Treating Duplicates
### Drop duplicates
df=df.drop_duplicates() <br>

df.info() <br>

### Drop NAs
df=df.dropna() <br>
df.info() <br>

### Count full address column and drop duplicates, keep the last entries.
full_address_counts = df['full_address'].value_counts() <br>
full_address_counts.head()

df=df.drop_duplicates(subset=['full_address'],keep= 'last') <br>
full_address_counts = df['full_address'].value_counts() <br>
full_address_counts.head() <br>
df.info() <br>

### Export data as Clean Data. This is Realtor_Data_Clean.csv
df.to_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean.csv',index=False) <br>
df=pd.read_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean.csv') <br>

## 1.2) Treating Outliers
### 1.2.1) Check and treat bed column
#### By grouping and counting, we can see that total bedrooms bigger than 10 are outliers, so we drop them.
df['bed'].describe() <br>
df.groupby('bed').agg({'price':'mean'}).plot.bar(legend=False) <br>
plt.ylabel('Avg Price') <br>
plt.xlabel('Total Number of Bedroom') <br>

![TotalNumberOfBedsGroup](https://user-images.githubusercontent.com/114496063/208707951-2c07f235-bed0-49fe-ba07-16046839ac80.png)

bed_counts = df['bed'].value_counts() <br>
bed_counts <br>

hist_columns=['price','bed','bath','house_size','acre_lot'] <br>
df[hist_columns[1]].hist() <br>
 
df.drop(df[(df['bed'] >10) | (df['bed'] < 1)].index, inplace=True) <br>
df.info() <br>

### 1.2.2) Check and treat bath column
#### By grouping and counting, we can see that total bathrooms bigger than 10 are outliers, so we drop them.
df['bath'].describe() <br>
df.groupby('bath').agg({'price':'mean'}).plot.bar(legend=False) <br>
plt.ylabel('Avg Price') <br>
plt.xlabel('Total Number of Bathroom') <br>

bath_counts = df['bath'].value_counts() <br>
bath_counts <br>
df[hist_columns[2]].hist() <br>
df.drop(df[(df['bath'] >10)].index, inplace=True) <br>
df.info() <br>


### 1.2.3) Check and treat house_size column
#### By grouping and counting, we can see that total house sizes bigger than 10000 are outliers, so we drop them.
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

### 1.2.4) Check and treat acre_lot column
#### By grouping and counting and checkimg percentiles, we can see that total acre lots bigger than 7.59 are outliers, so we drop them.
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

## 1.2.5) Check and treat price column
#### By grouping and counting and checkimg percentiles, we can see that prices bigger than 3.295M are outliers, so we drop them.

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
