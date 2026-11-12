from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.linear_model import LogisticRegression





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

    model_LR=Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000,
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

    joblib.dump(
                    model_LR,
                    "/opt/airflow/models/modelisation/Logistic_Regression.joblib"
    )
