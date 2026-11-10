from sklearn.svm import SVC
from src import preprocess




def Modelisation():

    X_train, X_test, y_train, y_test, model_SVC, model_RF, model_GB = preprocess.preprocess()

    models = {
        "SVC": model_SVC,
        "Random Forest": model_RF,
        "Gradient Boosting": model_GB
    }

    trained_models = {}

    for name, model in models.items():

        # Training
        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} trained successfully")

    return X_train, X_test, y_train, y_test, trained_models