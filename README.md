# Ex-07-Feature-Selection

# AIM:
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation:
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM:
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

## CODE:
```
NAME : DELLI PRIYA L
REGISTER NO. : 212222230029

# DATA PREPROCESSING BEFORE FEATURE SELECTION:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()

#checking data
df.isnull().sum()

#removing unnecessary data variables
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()

#cleaning data
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

#removing outliers 
plt.title("Dataset with outliers")
df.boxplot()
plt.show()

cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()

from sklearn.preprocessing import OrdinalEncoder
climate = ['male','female']
en= OrdinalEncoder(categories = [climate])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()

from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()

import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()

import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"] 

# FEATURE SELECTION:
# FILTER METHOD:
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()

# HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features

# BACKWARD ELIMINATION:
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

# RFE (RECURSIVE FEATURE ELIMINATION):
model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

# OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

# FINAL SET OF FEATURE:
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

# EMBEDDED METHOD:
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```

## OUTPUT:

### DATA PREPROCESSING BEFORE FEATURE SELECTION:
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/665adf58-8e75-4bb5-92a8-30f07e3d2f12)
![image](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/54ca6909-d903-4d8c-a143-1991d1b56640)
![241123224-ec725a93-f203-47a3-b138-4193215eb0aa](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/71aa98f3-4c59-4f0b-8d45-c5eaacd5093c)
![241123308-5cd3c41e-32a8-43db-89e8-2cdf0bc9095a](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/df538f6c-0427-4057-838f-12cbbb53a65d)
![241123360-17ec9127-4d7a-4a18-a70b-4930b84a089e](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/97e0d3a5-70b5-49f1-9bab-5d5917d18592)
![241123466-eaec3cc8-b2b6-4e5b-828d-5f54c22a81e1](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/822f24ac-ac74-4a3e-92d7-dd3e4f922f29)
![241123534-e990c8cd-a178-4132-9916-d80a285b9d4b](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/829ed3da-4c08-4043-93ce-c20ba0273f19)

### FEATURE SELECTION:
### FILTER METHOD:
The filtering here is done using correlation matrix and it is most commonly done using Pearson correlation. 
![241123694-0b8d3ec1-3c18-4923-9fb1-bd4e46b017e3](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/1270d84a-42aa-4727-a5bf-d93ba58e0bf2)

HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED: 
![241123806-deeac637-ded9-447b-9bb8-30e24224b454](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/6fb7167c-0ec3-4b4d-98ff-b6eca2cf766c)

### HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
![241124038-77404167-a114-4320-926e-e6c70be0f9c2](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/cb5c48dc-484a-4ec0-beb3-3c7e53dabd85)

### WRAPPER METHOD:
Wrapper Method is an iterative and computationally expensive process but it is more accurate than the filter method.

There are different wrapper methods such as Backward Elimination, Forward Selection, Bidirectional Elimination and RFE.

### BACKWARD ELIMINATION:
![241124141-38a8f1f3-73e9-4f1f-b4f3-38e78ab1e1ad](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/b313b2ce-d343-4501-8c5f-cea5d62e32bf)

### RFE (RECURSIVE FEATURE ELIMINATION):
![241124207-012c0a7c-b9e8-46e3-98e6-2eb0325e0c2b](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/16141396-7f27-4130-8024-1e070a6c5bba)

### OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
![241124252-159a43e2-20f9-4baf-8894-3c92bdb75e67](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/862857f8-39d5-4ac5-9db1-9953b4c5e82a)

### FINAL SET OF FEATURE:
![241124292-164c4499-e07d-482d-b5d8-5c2ae06805b8](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/4981a2fd-a995-437a-b9be-816bf935cb7c)

### EMBEDDED METHOD:
![241124425-16afc0a8-b963-42a4-a834-1df346a089bf](https://github.com/Priya-Loganathan/ODD2023-Datascience-Ex-07/assets/121166075/bed0f49d-b387-4717-8c21-c3b311a755b3)

## RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
