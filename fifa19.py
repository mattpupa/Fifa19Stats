#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:02:21 2020

@author: Matt
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Read fifa stats csv file into a dataframe
df_fifa = pd.read_csv('/Users/Matt/Desktop/DataScience/Kaggle/Fifa19stats/fifa19.csv', na_values=' ')

# Reduce df_fifa to dataframe with columns that will be used for model
df_model = df_fifa[['ID', 'Age', 'Nationality','Overall'
                    ,'Preferred Foot', 'International Reputation', 'Weak Foot'
                    ,'Skill Moves', 'Work Rate', 'Body Type', 'Position'
                    ,'Crossing', 'Finishing', 'HeadingAccuracy'
                    ,'ShortPassing', 'Volleys', 'Dribbling', 'Curve'
                    ,'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration'
                    ,'SprintSpeed', 'Agility', 'Reactions', 'Balance'
                    ,'ShotPower','Jumping', 'Stamina', 'Strength', 'LongShots'
                    ,'Aggression','Interceptions', 'Positioning', 'Vision'
                    ,'Penalties', 'Composure', 'Marking', 'StandingTackle'
                    ,'SlidingTackle', 'GKDiving', 'GKHandling'
                    ,'GKKicking', 'GKPositioning', 'GKReflexes']].copy()

df_model.dropna(inplace=True)

# Create y_model now since we will drop 'R_750K' in df_model
y_model = df_model['Overall']

# Remove 'Overall' column since that is our y, and convert df_model to 
# indicator feature set and create X_model
df_model.drop(['Overall'],axis=1,inplace=True)
X_model = pd.get_dummies(df_model)

# Create scores list
scores_list = []


# Create testing and training data sets
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model
                                                    , random_state=0)

# Linear regression
linreg = LinearRegression().fit(X_train, y_train)
linreg_score_train = linreg.score(X_train, y_train)
linreg_score_test = linreg.score(X_test, y_test)

scores_list.append(['linear regression', np.nan, np.sum(linreg.coef_ != 0)
                    , linreg_score_train, linreg_score_test])


# Ridge regression with normalization

alpha_list = [0, 1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 1000]

# Fit and transform X_train and transform X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=3).fit(X_train_scaled, y_train)
linridge_scaled_score_train = linridge.score(X_train_scaled, y_train)
linridge_scaled_score_test = linridge.score(X_test_scaled, y_test)

def ridge_normalized_by_alpha_scaled():
    for a in alpha_list:
        ridge = Ridge(alpha=a).fit(X_train_scaled, y_train)
        ridge_scaled_score_train = ridge.score(X_train_scaled, y_train)
        ridge_scaled_score_test = ridge.score(X_test_scaled, y_test)
        scores_list.append(['ridge regression', a, np.sum(ridge.coef_ != 0)
                    , ridge_scaled_score_train, ridge_scaled_score_test])
        #print(a, ridge_scaled_score_train, ridge_scaled_score_test)

# Lasso regression
linlasso = Lasso(alpha=1, max_iter = 10000).fit(X_train_scaled, y_train)
linlasso_scaled_score_train = linlasso.score(X_train_scaled, y_train)
linlasso_scaled_score_test = linlasso.score(X_test_scaled, y_test)

def lasso_normalized_by_alpha_scaled():
    for a in alpha_list:
        lasso = Lasso(alpha=a).fit(X_train_scaled, y_train)
        lasso_scaled_score_train = lasso.score(X_train_scaled, y_train)
        lasso_scaled_score_test = lasso.score(X_test_scaled, y_test)
        scores_list.append(['lasso regression', a, np.sum(lasso.coef_ != 0)
                    , lasso_scaled_score_train, lasso_scaled_score_test])
        #print(a, lasso_scaled_score_train, lasso_scaled_score_test)



ridge_normalized_by_alpha_scaled()
lasso_normalized_by_alpha_scaled()



df_scores = pd.DataFrame(scores_list, columns=['model_type', 'alpha', 'features_kept'
                                               ,'r2_train', 'r2_test'])

y_predictions = list(linridge.predict(X_test_scaled))
idlist = list(X_test['ID'])

df_predictions = pd.DataFrame(list(zip(idlist, y_predictions)), columns=['ID', 'Prediction'])

df_predictions_merged = df_predictions.merge(df_fifa, how='inner', left_on='ID',right_on='ID')

df_final = df_predictions_merged[['Name', 'Overall', 'Prediction']]



