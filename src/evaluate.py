
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


def evaluation():

    path = "/opt/airflow/data/X_train.csv"

    print("Exists:", os.path.exists(path))
    print("Size:", os.path.getsize(path))

    with open(path, "r") as f:
        print(f.read(500))

    # 1. Load training and testing data
    X_train = pd.read_csv("/opt/airflow/data/X_train.csv")
    X_test = pd.read_csv("/opt/airflow/data/X_test.csv")

    y_train = pd.read_csv( "/opt/airflow/data/y_train.csv").squeeze("columns")

    y_test = pd.read_csv("/opt/airflow/data/y_test.csv").squeeze("columns")


    # 2. Load existing Pipelines
    model_svc = joblib.load("/opt/airflow/models/modelisation/SVC.joblib")

    model_rf = joblib.load("/opt/airflow/models/modelisation/Random_Forest.joblib")

    model_gb = joblib.load("/opt/airflow/models/modelisation/Gradient_Boosting.joblib")

    # 3. Define models and hyperparameters
    models = {
        "SVC": {
            "model": model_svc,
            "params": [
                {
                    "classifier__C": [0.1, 1, 10],
                    "classifier__kernel": ["linear"]
                },
                {
                    "classifier__C": [0.1, 1, 10],
                    "classifier__kernel": ["rbf"],
                    "classifier__gamma": ["scale", "auto"]
                }
            ]
        },

        "Random Forest": {
            "model": model_rf,
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10],
                "classifier__min_samples_split": [2, 5]
            }
        },

        "Gradient Boosting": {
            "model": model_gb,
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3]
            }
        }
    }

    # 4. Cross Validation configuration
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = {}
    best_models = {}

    # 5. GridSearchCV for each model
    for name, config in models.items():

        print(f"\n{name} ")

        # Verify pipeline steps
        # print("Pipeline steps:", config["model"].named_steps.keys())

        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
            verbose=1,
            refit=True
        )

        grid_search.fit(X_train, y_train)

        results[name] = {
            "best_cv_score": float(grid_search.best_score_),
            "best_params": grid_search.best_params_
        }

        best_models[name] = grid_search.best_estimator_

        print("Best Parameters:", grid_search.best_params_)
        print("Best CV F1 Macro:", grid_search.best_score_)

    # 6. Compare all models
    comparison = pd.DataFrame([
        {
            "Model": name,
            "CV F1 Macro": result["best_cv_score"]
        }
        for name, result in results.items()
    ])

    comparison = comparison.sort_values(
        by="CV F1 Macro",
        ascending=False
    )

    print("\nMODEL COMPARISON:")
    print(comparison.to_string(index=False))

    # 7. Select the best model
    best_name = comparison.iloc[0]["Model"]
    best_model = best_models[best_name]

    print("\nBest Model:", best_name)

    # 8. Final evaluation on test set
    y_pred = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nFINAL TEST EVALUATION")
    print("Test Accuracy:", test_accuracy)
    print("Test F1 Macro:", test_f1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 9. Save best model and results
 
       # 9. Save best model and results

    output_dir = "/opt/airflow/Models/Best"

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.joblib")
    json_path = os.path.join(output_dir, "best_model.json")

    # Save model
    joblib.dump(best_model, model_path)

    # Save results
    Best = {
        "best_model": best_name,
        "best_cv_f1_macro": float(results[best_name]["best_cv_score"]),
        "best_params": results[best_name]["best_params"],
        "test_accuracy": float(test_accuracy),
        "test_f1_macro": float(test_f1)
    }

    with open(json_path, "w") as file:
        json.dump(Best, file, indent=4)

    # Verify saving
    print("\nSAVING VERIFICATION")

    print("Model path:", model_path)
    print("Model exists:", os.path.exists(model_path))
    print("JSON exists:", os.path.exists(json_path))
    print("Files:", os.listdir(output_dir))