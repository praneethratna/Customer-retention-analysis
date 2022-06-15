import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('customer-churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
df.drop(['customerID'], inplace = True, axis = 1)
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace = True)

#Creating dummy variables for categorical variables
df['Dependents'] = pd.get_dummies(df['Dependents'], drop_first=True)
df['Churn'] = pd.get_dummies(df['Churn'], drop_first = True)
df['Partner'] = pd.get_dummies(df['Partner'], drop_first=True)
df['gender'] = pd.get_dummies(df['gender'], drop_first = True)
df['SeniorCitizen'] = pd.get_dummies(df['SeniorCitizen'], drop_first=True)
df['PhoneService'] = pd.get_dummies(df['PhoneService'], drop_first = True)
df['PaperlessBilling'] = pd.get_dummies(df['PaperlessBilling'], drop_first = True)
s1 = ['MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection',
'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, drop_first = True, columns = s1)

#Dropping total charges column
df.drop('TotalCharges', inplace = True, axis = 1)

#Splitting data into dependent and independent variables
X = df.drop(labels = 'Churn', axis = 1)
y = df['Churn']

#Splitting data into train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Best parameters obtained from GridSearchCV
rfc = RandomForestClassifier(n_estimators = 130, max_depth = 8)
rfc.fit(X_train, y_train)

pickle.dump(rfc, open("model.pkl", "wb"))