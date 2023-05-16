# Ex-06-Feature-Transformation
AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

ALGORITHM:
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

PROGRAM:
```
Developed by: Rubika A
Register No. : 212220220035
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
```
```
df.head()
df.isnull().sum()
df.info()
df.describe()
```
```
df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
```
sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()
```
```
sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()
```
```
sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()
```
```
df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
```
df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
```
df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
```
df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()
```
```
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()
```
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```

OUTPUT:

![dataex6](https://user-images.githubusercontent.com/118680410/234250308-3b26b5f3-f092-4d5b-8fff-e396184d7db5.png)
![dataex6 1](https://user-images.githubusercontent.com/118680410/234249807-8304c912-22f5-4247-a0bc-5b4fe5cbf985.png)
![dataex6 2](https://user-images.githubusercontent.com/118680410/234249913-e6ddc5d2-bf49-4267-a69a-2b3141be8d9d.png)
![dataex6 3](https://user-images.githubusercontent.com/118680410/234249938-a38d4c81-815a-48da-a498-38dbdab17ca6.png)
![dataex6 4](https://user-images.githubusercontent.com/118680410/234250047-e9ffad02-2570-434b-88e1-829d0ea3498c.png)
![dataex6 5](https://user-images.githubusercontent.com/118680410/234250126-4ff0d4f1-16e2-4705-af9e-ae25d73e72f8.png)
![dataex6 8](https://user-images.githubusercontent.com/118680410/234250181-c07a01c1-c879-4508-9443-7a0c5fb25839.png)
![dataex6 9](https://user-images.githubusercontent.com/118680410/234250190-171ad543-2cca-48fe-8261-b07a390c3088.png)
![dataex6 6](https://user-images.githubusercontent.com/118680410/234250194-c9cf4eff-b3c0-4052-bb24-f29c3c0a7683.png)
![dataex6 10](https://user-images.githubusercontent.com/118680410/234250238-954db167-9783-4fa4-b723-03eabd13d5c5.png)
![dataex6 11](https://user-images.githubusercontent.com/118680410/234250275-bfd6b002-ad2b-443d-8706-4e5b309ca8b0.png)

RESULT:

Thus Feature transformation is performed and executed successfully for the given dataset




