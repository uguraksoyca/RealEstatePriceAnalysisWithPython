
# RealEstatePriceAnalysisWithPython
Dataset : https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset <br>

import pandas as pd <br>
import numpy as np <br>
import matplotlib.pyplot as plt <br>
import matplotlib as mpl <br>
mpl.rcParams['figure.dpi']=400 <br> 
from sklearn.linear_model import LinearRegression <br>

pd.options.display.max_columns = None <br>

## Read file, Check the data, Note that you need to point the Excel reader to wherever the file is located.
df=pd.read_csv('C:/WorkSamples/HousePricePrediction/realtor-data.csv') <br>
df.head() <br>
df.info() <br>

# 1) DATA VALIDATION  <br>
## 1.1) Treating Duplicates
### Drop duplicates
df=df.drop_duplicates() <br>

df.info() <br>

![DFInfo](https://user-images.githubusercontent.com/114496063/208709112-c59fab1c-b4bd-47cc-8f54-6377827b6b4d.png)

### Drop NAs
#### There are many NA values. Instead of implementing any values, we choose to drop them.
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

![BedCounts](https://user-images.githubusercontent.com/114496063/208708644-3d82916c-cadb-4e95-bf92-f83d4d6e6981.png)

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

![TotalNumberOfBathroomGroup](https://user-images.githubusercontent.com/114496063/208707906-61c13ebc-64a7-4eec-a89a-2ea70461d0b2.png)

bath_counts = df['bath'].value_counts() <br>
bath_counts <br>

![BathCounts](https://user-images.githubusercontent.com/114496063/208710432-5957775b-7755-4e9a-a858-1cb4374b9eb9.png)

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

![AcreLotPercentile](https://user-images.githubusercontent.com/114496063/208707804-bd11be21-c9ab-4834-98a4-50c1646c590a.png)

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

![PricePercentile](https://user-images.githubusercontent.com/114496063/208707840-27b488ca-051f-4718-909b-ec1fb16050e6.png)

df.drop(df[(df['price(M)'] >3.295)].index, inplace=True) <br>
df.info() <br>

df.to_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean_Without_Outliers.csv',index=False) <br>

# 2-) REGRESSION ANALYSIS  <br>

from sklearn.model_selection import train_test_split <br>
from sklearn.linear_model import LinearRegression <br> 
from sklearn.ensemble import RandomForestRegressor <br>
from sklearn.metrics import mean_absolute_error, r2_score <br>

corrM = df.corr() <br>
corrM

### We can see that bath and house_size has more that 0.6 correlation with price. We can use these variables to create linear regression models. <br>

![Correlation Matrix](https://user-images.githubusercontent.com/114496063/208973139-c115c710-25cc-4dba-b13b-e628bc28c0ee.png)

## 2.1) Single Linear Regression  <br>
df=pd.read_csv('C:/WorkSamples/HousePricePrediction/Realtor_Data_Clean_Without_Outliers.csv') <br>

## 2.1.1) Single Linear Regression - Bath and Price  <br>
X_train,X_Test,y_train,y_test=train_test_split(df['bath'].values.reshape(-1,1),  <br>                                           df['price'].values,test_size=0.2,random_state=24) <br>

model = LinearRegression() <br>

model.fit(X_train, y_train)  <br>
model_pred=model.predict(X_Test) <br>

mae = mean_absolute_error(y_test, model_pred) <br>
r2 = r2_score(y_test, model_pred) <br>
r2_percent = r2*100 <br>

print(f"The accuracy of price based on bath predictor is: {r2_percent:.2f}%")  <br>

### The accuracy of price based on bath predictor is: 38.92% <br>
 
print(f"On average, this listing price based on bath predictor is $ {mae:.2f} off of any given price.") <br>

### On average, this listing price based on bath predictor is $ 249138.27 off of any given price. <br>

## 2.1.2) Single Linear Regression - House Size and Price  <br>

X_train2,X_Test2,y_train2,y_test2=train_test_split(df['house_size'].values.reshape(-1,1), <br>
df['price'].values,test_size=0.2,random_state=24) <br>
model2 = LinearRegression() <br>
model2.fit(X_train2, y_train2) <br>
model_pred2=model2.predict(X_Test2) <br>
mae_2 = mean_absolute_error(y_test2, model_pred2) <br>
r2_2 = r2_score(y_test2, model_pred2) <br>
r2_percent_2 = r2_2*100 <br>

print(f"The accuracy of price based on house_size predictor is: {r2_percent_2:.2f}%") <br>

### The accuracy of price based on house_size predictor is: 36.83% <br>

print(f"On average, this listing price based on house_size predictor is $ {mae_2:.2f} off of any given price.") <br>

### On average, this listing price based on house_size predictor is $ 247080.12 off of any given price. <br>

## 2.2) Multi Linear Regression  <br>

x3 = df[['bed', 'bath', 'acre_lot','house_size']] <br>
y3 = df['price'] <br>

x_train3, x_test3,y_train3, y_test3 = train_test_split(x3, y3, test_size = 0.2, random_state = 100) <br>
model3 = LinearRegression()   <br>
model3.fit(x_train3, y_train3) <br>
model_pred3= model3.predict(x_test3) <br>
mae_3 = mean_absolute_error(y_test3, model_pred3) <br>
r2_3 = r2_score(y_test3, model_pred3) <br>
r2_percent_3 = r2_3*100 <br>
print(f"The accuracy of price based on house_size predictor is: {r2_percent_3:.2f}%") <br>

### The accuracy of price based on house_size predictor is: 42.66% <br>

print(f"On average, this listing price based on house_size predictor is $ {mae_3:.2f} off of any given price.") <br>

### On average, this listing price based on house_size predictor is $ 235160.45 off of any given price. <br>

## 2.3) Random Forest Regression  <br>
## 2.3.1) Random Forest Regression - Bath and Price  <br>

x4 = df.bath.values.reshape(-1, 1)
y4 = df.price.values
x_train4, x_test4,y_train4, y_test4 = train_test_split(x4, y4, test_size = 0.2, random_state = 40) <br>
model4 = RandomForestRegressor() <br>
model4.fit(x_train4, y_train4) <br>
model_pred4 = model4.predict(x_test4) <br>
mae_4 = mean_absolute_error(y_test4, model_pred4) <br>
r2_4 = r2_score(y_test4, model_pred4) <br>
r2_percent_4 = r2_4*100 <br>
print(f"The accuracy of price based on bath predictor is: {r2_percent_4:.2f}%") <br>

### The accuracy of price based on bath predictor is: 42.55% <br>

print(f"On average, this listing price based on bath predictor is $ {mae_4:.2f} off of any given price.") <br>

### On average, this listing price based on bath predictor is $ 237871.41 off of any given price. <br> 

## 2.3.2) Random Forest Regression - House Size and Price  <br>

x5 = df.house_size.values.reshape(-1, 1) <br>
y5 = df.price.values <br>
x_train5, x_test5,y_train5, y_test5 = train_test_split(x5, y5, test_size = 0.2, random_state = 40) <br>
model5 = RandomForestRegressor() <br>
model5.fit(x_train5, y_train5) <br>
model_pred5 = model5.predict(x_test5) <br>
mae_5 = mean_absolute_error(y_test5, model_pred5) <br>
r2_5 = r2_score(y_test5, model_pred5) <br>
r2_percent_5 = r2_5*100 <br>

print(f"The accuracy of price based on house_size predictor is: {r2_percent_5:.2f}%") <br>

### The accuracy of price based on house_size predictor is: 28.27% <br>

print(f"On average, this listing price based on bath predictor is $ {mae_5:.2f} off of any given price.") <br>

### On average, this listing price based on bath predictor is $ 261569.96 off of any given price.

## 2.3.3) Random Forest Regression - All Columns  <br>

X6 = df.drop("price", axis = 1) <br>
y6 = df["price"] <br>
X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, test_size = 0.2) <br>
model6 = RandomForestRegressor() <br>
model6.fit(X_train6.values, y_train6) <br>
model_pred6 = model6.predict(X_test6.values) <br>
model_pred6_diff = pd.DataFrame({'Actual value': y_test6, 'Predicted value': model_pred6}) <br>
model_pred6_diff.head() <br>

print(f"The accuracy of price based on house_size predictor is: {r2_percent_6:.2f}%") <br>

### The accuracy of price based on house_size predictor is: 71.74% <br>

print(f"On average, this listing price based on bath predictor is $ {mae_6:.2f} off of any given price.") <br>

### On average, this listing price based on bath predictor is $ 140451.26 off of any given price. <br>

results = pd.DataFrame({'SL_Bath_and_Price': [mae, r2, r2_percent], <br>
                   'SL_House_Size_and_Price': [mae_2, r2_2, r2_percent_2], <br>
                   'ML': [mae_3, r2_3, r2_percent_3], <br>
                   'RF_Bath_and_Price': [mae_4, r2_4, r2_percent_4], <br>
                   'RF_House_Size_and_Price': [mae_5, r2_5, r2_percent_5], <br>
                   'RF_All_Columns_For_Price': [mae_6, r2_6, r2_percent_6]}, <br>
                  index=['mean_absolute_error', 'r2_score', 'r2_percent']) <br>
                  
results
![results](https://user-images.githubusercontent.com/114496063/208959873-ac643742-921f-4d20-a9ec-68e68e436f30.png)

## As we can see in the results dataframe, Random Forest regression model which is model 6 is the best model.
