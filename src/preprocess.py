import pandas as pd
from src import eda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import math
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier






def preprocess():
    X,y,num_cols,nominal_cols,ordinal_cols,df=eda.EDA()
    
    result = df[
    df["Social_Media_Hours"] > df["Screen_Time_Hours"]
    ]

    # print("Number of rows to remove:", len(result))

    # print(result[
    #     ["Social_Media_Hours", "Screen_Time_Hours"]
    # ])

    # Remove rows
    df = df.drop(result.index)

    # print("Remaining rows:", len(df))

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)




    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=[
                ['Low', 'Moderate', 'High'],
                ['Poor', 'Average', 'Good']
            ]
        ))
    ])

    preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('nominal', nominal_pipeline, nominal_cols),
    ('ordinal', ordinal_pipeline, ordinal_cols)
    ])


    model_SVC = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())
    ])

    model_RF = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
    ])
    model_GB = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
    ])

    return X_train,X_test,y_train,y_test,model_SVC,model_RF,model_GB