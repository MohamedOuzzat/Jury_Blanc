import pandas as pd
from src import eda
import numpy as np
import matplotlib.pyplot as plt
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
import joblib

def preprocess():
    df = pd.read_csv("/opt/airflow/data/student_mental_health.csv")
  
    metadata=eda.EDA()
  

    num_cols=metadata["num_cols"]
    nominal_cols=metadata["nominal_cols"]
    ordinal_cols=metadata["ordinal_cols"]
    target_col=metadata["target_col"]

    target_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
    }

    df[target_col] = df[target_col].map(target_mapping)

    
    result = df[
    df["Social_Media_Hours"] > df["Screen_Time_Hours"]
    ]



    # Remove rows
    df = df.drop(result.index)

    X = df.drop(columns=["Student_ID", "Mental_Health_Risk"])
    y = df[target_col]
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
                ["Low", "Moderate", "High"],
                ["Poor", "Average", "Good"]
            ]
        ))
    ])

    preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('nominal', nominal_pipeline, nominal_cols),
    ('ordinal', ordinal_pipeline, ordinal_cols)
    ])

    X_train.to_csv("/opt/airflow/data/X_train.csv",index=False)
    X_test.to_csv("/opt/airflow/data/X_test.csv",index=False)
    y_train.to_csv("/opt/airflow/data/y_train.csv",index=False)
    y_test.to_csv("/opt/airflow/data/y_test.csv",index=False)

   

    joblib.dump(
        preprocessor,
        "/opt/airflow/data/processed/preprocessor.joblib"
    )

