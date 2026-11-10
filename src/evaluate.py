import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src import preprocess
from src import modelisation


def Evaluation():

    X_train, X_test, y_train, y_test, models = modelisation.Modelisation()

    results = []

    for name, model in models.items():

        # 1. Cross Validation
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=5,
            scoring='accuracy'
        )

        scores = cv_results['test_score']

        # 2. Prediction
        y_pred = model.predict(X_test)

        # 3. Evaluation
        results.append({
            "Model": name,
            "Mean": scores.mean(),
            "Std": scores.std(),
            "Min": scores.min(),
            "Max": scores.max(),
            "Test Accuracy": accuracy_score(y_test, y_pred)
        })


        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # 4. Compare models
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Mean",
        ascending=False
    )

    print("\n Models Comparison :")
    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]

    print(f"\nBest Model: {best_model_name}")

    return results_df