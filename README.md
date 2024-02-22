# EXP-02  Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SRI SAI PRIYA.S
RegisterNumber:  212222240103
*/


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)
#Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
1.df.head()
![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475702/1dc2ddc5-0ded-41c0-9b18-1cc962bf5a98)

2.df.tail()
![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475702/9eada9f6-bd10-46f0-af95-a04fe41fd553)

3.Array Value of X
![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475702/b2a12855-b023-4c72-87b6-daa8492d5a51)

4.Array Value of Y
![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475702/1f47ce40-7c1a-4c87-a143-b77d8e79e7e3)

5.Values of Y prediction
![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475702/500731ac-efa7-4104-9a26-124cf596ed5a)

6.Array values of Y test
![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475702/8f60ae89-df39-422d-a788-2e8eddd2a6ed)

7.Training Set Graph
![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475702/b2bcb3f8-e9f6-4895-aa2b-96697b99c70a)

8.Test Set Graph
![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475702/65344987-511d-417a-b8c6-dd4923729b66)

9.Values of MSE, MAE and RMSE
![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475702/8b2b0a33-5bd5-4b4c-b0d7-e80b536334c3)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
