from sklearn.svm import SVC
from src import preprocess
from sklearn.pipeline import Pipeline
from src import preprocess
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import joblib




def modelisation():

    preprocessor=joblib.load("/opt/airflow/data/processed/preprocessor.joblib")

    model_SVC = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True))
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
            random_state=42
        ))
    ])

    joblib.dump(
            model_SVC,
            "/opt/airflow/models/modelisation/SVC.joblib"
    )
    joblib.dump(
                model_RF,
                "/opt/airflow/models/modelisation/Random_Forest.joblib"
    )
    joblib.dump(
                model_GB,
                "/opt/airflow/models/modelisation/Gradient_Boosting.joblib"
    )
