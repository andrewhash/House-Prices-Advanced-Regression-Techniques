#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:32:50 2024

@author: andrewhashoush
"""

import pandas as pd
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
import sklearn.tree
import sklearn.ensemble

df = pd.read_csv('/Users/andrewhashoush/Downloads/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/Users/andrewhashoush/Downloads/house-prices-advanced-regression-techniques/test.csv')

print(df.head())
print(df.info())

df['Quality_Condition'] = df['OverallQual'] * df['OverallCond']
df['Age_at_Sale'] = df['YrSold'] - df['YearBuilt']

df_test['Quality_Condition'] = df_test['OverallQual'] * df_test['OverallCond']
df_test['Age_at_Sale'] = df_test['YrSold'] - df_test['YearBuilt']

selected_features = ['Id','OverallQual', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea',
                    'GarageCars', 'GarageArea', 'MSZoning', 'Neighborhood',
                    'KitchenQual', 'CentralAir', 'LotArea', 'MSSubClass', 'LotFrontage', 
                    'Street', 'LandContour', 'OverallCond', 'RoofStyle',
                     'BsmtQual','SaleCondition', 'SaleType', 'YrSold', 'MoSold', 
                    'PoolArea','BedroomAbvGr','Foundation','Quality_Condition','Age_at_Sale']

X = df[selected_features]
df_test = df_test[selected_features]

for column in X.columns:
    missing = X[column].isnull().sum()
    print(f"{column}: {missing}")
    
numerical = ['LotFrontage', 'OverallQual','YearBuilt','TotalBsmtSF', '1stFlrSF','GrLivArea', 'GarageCars',
             'GarageArea', 'LotArea','MSSubClass','LotFrontage', 'OverallCond', 'YrSold', 'MoSold'
             ,'PoolArea','BedroomAbvGr', 'Quality_Condition','Age_at_Sale' ]

categorical = ['MSZoning','Neighborhood','KitchenQual', 'CentralAir', 'Street', 'LandContour','RoofStyle'
                , 'BsmtQual','SaleCondition', 'SaleType','Foundation' ]

for var in categorical:
    mode_val = X[var].mode()[0]
    X[var] = X[var].fillna(mode_val)
    df_test[var] = df_test[var].fillna(mode_val)
    
for var in numerical:
    median_val = X[var].median()
    X[var] = X[var].fillna(median_val)
    df_test[var] = df_test[var].fillna(median_val)

    
for column in X.columns:
     missing = X[var].isnull().sum()
     print(f"{column}: {missing}")

for column in df_test.columns:
    missing = df_test[column].isnull().sum()
    print(f"{column}: {missing}")
    
X = pd.get_dummies(X, columns= categorical)
df_test = pd.get_dummies(df_test, columns = categorical)

y = df['SalePrice']

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X,y, train_size =.8, random_state= 123)

dt_model = sklearn.tree.DecisionTreeRegressor(max_depth=7, random_state=123)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_val)  
mse_val = np.mean((y_pred - y_val)**2)
print(f"MSE: {mse_val}")

# Random Forest Regressor
rf_model = sklearn.ensemble.RandomForestRegressor(n_estimators=500, random_state=123)
rf_model.fit(X_train, y_train)
y_pred1 = rf_model.predict(X_val)
mse_val1 = np.mean((y_pred1 - y_val)**2)
print(f"MSE: {mse_val1}")

# Gradient Boosting Regressor (my best Regressor)
gb_model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=300, learning_rate=.1, random_state=123)
gb_model.fit(X_train, y_train)
y_pred2 = gb_model.predict(X_val)
mse_val2 = np.mean((y_pred2 - y_val)**2)
print(f"MSE: {mse_val2}")


final_pred = gb_model.predict(df_test)
sub = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': final_pred})
sub.to_csv('/Users/andrewhashoush/Downloads/house-prices-advanced-regression-techniques/sample_submission.csv', index=False)
    
    
    
    
    
    
    
    
    
    