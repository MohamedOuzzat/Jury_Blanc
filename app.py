
import os
import json
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from sklearn.preprocessing import label_binarize


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Mental Health AI",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "Models", "Best", "best_model.joblib"
)

JSON_PATH = os.path.join(
    BASE_DIR, "Models", "Best", "best_model.json"
)

X_TEST_PATH = os.path.join(
    BASE_DIR, "data", "X_test.csv"
)

Y_TEST_PATH = os.path.join(
    BASE_DIR, "data", "y_test.csv"
)

LABELS = {
    0: "Low",
    1: "Medium",
    2: "High"
}

CLASS_IDS = [0, 1, 2]
CLASS_NAMES = ["Low", "Medium", "High"]


# ==========================================
# 2. LOAD MODEL AND DATA
# ==========================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_test_data(x_path, y_path):
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze("columns")

    return X_test, y_test


@st.cache_data
def load_model_info(path):
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)

    return {}


# ==========================================
# 3. CHECK FILES
# ==========================================

st.title("🧠 Student Mental Health Prediction")

required_files = [
    MODEL_PATH,
    X_TEST_PATH,
    Y_TEST_PATH
]

missing_files = [
    path for path in required_files
    if not os.path.isfile(path)
]

if missing_files:
    st.error("Some required files are missing:")

    for path in missing_files:
        st.write(path)

    st.stop()

try:
    model = load_model(MODEL_PATH)

    X_test, y_test = load_test_data(
        X_TEST_PATH,
        Y_TEST_PATH
    )

    model_info = load_model_info(JSON_PATH)

except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Model Performance",
        "Test New Data"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

best_model_name = model_info.get(
    "best_model",
    "Best Model"
)

st.sidebar.write("Best Model:", best_model_name)

st.sidebar.write(
    "CV F1 Macro:",
    f"{model_info.get('best_cv_f1_macro', 0):.2%}"
)


# ==========================================
# 5. MODEL PERFORMANCE
# ==========================================

if page == "Model Performance":

    st.header("📊 Model Performance")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Macro", f"{f1:.2%}")

    st.divider()

    # Airflow comparison
    st.subheader("Airflow vs Streamlit")

    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Macro"],
        "Airflow": [
            model_info.get("test_accuracy", 0),
            model_info.get("test_f1_macro", 0)
        ],
        "Streamlit": [
            accuracy,
            f1
        ]
    })

    st.dataframe(
        comparison_df.style.format({
            "Airflow": "{:.2%}",
            "Streamlit": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(7, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=CLASS_IDS,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    ax.set_title("Confusion Matrix - " + best_model_name)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Classification Report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df["support"] = report_df["support"].astype(int)

    st.dataframe(
        report_df.style.format({
            "precision": "{:.2%}",
            "recall": "{:.2%}",
            "f1-score": "{:.2%}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

    st.divider()

    # Performance by Class
    st.subheader("Performance by Class")

    class_df = report_df.loc[
        CLASS_NAMES,
        ["precision", "recall", "f1-score"]
    ]

    st.bar_chart(class_df)

    st.divider()

    # ROC Curve
    st.subheader("ROC Curve")

    if hasattr(model, "predict_proba"):

        y_proba = model.predict_proba(X_test)

        y_test_bin = label_binarize(
            y_test,
            classes=model.classes_
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, class_label in enumerate(model.classes_):

            RocCurveDisplay.from_predictions(
                y_test_bin[:, i],
                y_proba[:, i],
                name=LABELS.get(int(class_label), str(class_label)),
                ax=ax
            )

        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            label="Random Classifier"
        )

        ax.set_title("ROC Curve - " + best_model_name)

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("ROC Curve requires a model supporting predict_proba().")

    st.caption(
        "All performance metrics are calculated using the saved test dataset."
    )


# ==========================================
# 6. TEST NEW DATA
# ==========================================

elif page == "Test New Data":

    st.header("🔍 Test the Model on New Data")

    st.write(
        "Upload a CSV file containing new student records "
        "with the same features used during model training."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            new_data = pd.read_csv(uploaded_file)

            # Remove target if included
            new_data = new_data.drop(
                columns=["Mental_Health_Risk"],
                errors="ignore"
            )

            expected_columns = list(X_test.columns)

            missing_columns = [
                col for col in expected_columns
                if col not in new_data.columns
            ]

            extra_columns = [
                col for col in new_data.columns
                if col not in expected_columns
            ]

            if missing_columns:
                st.error(
                    f"Missing features: {missing_columns}"
                )
                st.stop()

            if extra_columns:
                st.warning(
                    f"Extra columns will be ignored: {extra_columns}"
                )

            # Keep correct features and order
            new_data = new_data[expected_columns]

            st.subheader("Uploaded Data")

            st.dataframe(
                new_data,
                use_container_width=True
            )

            if st.button("Predict Mental Health Risk"):

                predictions = model.predict(new_data)

                results = new_data.copy()

                results["Predicted_Risk"] = [
                    LABELS.get(int(pred), str(pred))
                    for pred in predictions
                ]

                st.subheader("Prediction Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                # Prediction distribution
                st.subheader("Prediction Distribution")

                distribution = (
                    results["Predicted_Risk"]
                    .value_counts()
                    .reindex(
                        CLASS_NAMES,
                        fill_value=0
                    )
                )

                st.bar_chart(distribution)

                # Download results
                st.download_button(
                    "Download Predictions CSV",
                    data=results.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="mental_health_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ==========================================
# 7. DISCLAIMER
# ==========================================

st.divider()

st.warning(
    "Disclaimer: This application is an educational machine "
    "learning prototype. Its predictions are not a clinical "
    "diagnosis and should not replace professional assessment."
)
