
import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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

# ==========================================
# 2. LOAD MODEL AND DATA
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

    return X_test, y_test


@st.cache_data
def load_model_info():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as file:
            return json.load(file)

    return {}


# ==========================================
# 3. CHECK FILES
# ==========================================

st.title("🧠 Student Mental Health Prediction")
st.caption("Machine Learning Dashboard | SVC Model")

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
    model = load_model()
    X_test, y_test = load_test_data()
    model_info = load_model_info()

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

st.sidebar.write(
    "Best Model:",
    model_info.get("best_model", "SVC")
)

st.sidebar.write(
    "CV F1 Macro:",
    f"{model_info.get('best_cv_f1_macro', 0):.2%}"
)


# ==========================================
# 5. MODEL PERFORMANCE
# ==========================================

if page == "Model Performance":

    st.header("📊 Model Performance")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Macro", f"{f1:.2%}")

    st.divider()

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])

    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticklabels(["Low", "Medium", "High"])

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                color="black"
            )

    fig.colorbar(image, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Classification Report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.style.format({
            "precision": "{:.2%}",
            "recall": "{:.2%}",
            "f1-score": "{:.2%}"
        }),
        use_container_width=True
    )

    # Performance by class
    st.subheader("Performance by Class")

    class_df = report_df.loc[
        ["Low", "Medium", "High"],
        ["precision", "recall", "f1-score"]
    ]

    st.bar_chart(class_df)

    st.caption(
        "All metrics are calculated using the saved test dataset."
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

            # Remove target if included in uploaded file
            new_data = new_data.drop(
                columns=["Mental_Health_Risk"],
                errors="ignore"
            )

            # Expected features
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

            # Keep only required columns in correct order
            new_data = new_data[expected_columns]

            st.subheader("Uploaded Data")
            st.dataframe(new_data, use_container_width=True)

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
                        ["Low", "Medium", "High"],
                        fill_value=0
                    )
                )

                st.bar_chart(distribution)

                st.download_button(
                    "Download Predictions CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
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
