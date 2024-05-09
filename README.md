# Ex No:9 Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sanjay M
RegisterNumber:  212222110038
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

## Result Output:

![image](https://github.com/Jai-1801/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139335300/7c5f461a-d8d1-41c3-9852-a5041e1c2b61)

## data.head():

![image](https://github.com/Jai-1801/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139335300/32b859d4-a18b-4785-8aad-1238c25b6305)

## data.info():

![image](https://github.com/Jai-1801/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139335300/0f7f3b8e-53c5-4d4e-ae7c-192ff7d9bbdd)

## Y_prediction value:

![image](https://github.com/Jai-1801/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139335300/40a5f99f-4cd1-4b5b-842d-504d1be0c478)

## Accuracy value :

![image](https://github.com/Jai-1801/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139335300/2005b5c8-614e-454f-8f2d-31ef60c3c3d1)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
