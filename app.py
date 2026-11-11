
import os
import joblib
import pandas as pd
import streamlit as st

# Model path
MODEL_PATH = "models/Best/best_model.joblib"

# Page configuration
st.set_page_config(
    page_title="Student Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

# Load model once
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Title
st.title("🧠 Student Mental Health Risk Prediction")
st.write(
    "Enter the student's information to predict their mental health risk level."
)

# Check model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

model = load_model()

# Input form
with st.form("prediction_form"):

    st.subheader("👤 Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 15, 60, 20)
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        year = st.selectbox("Year of Study", [1, 2, 3, 4, 5])
        part_time = st.selectbox("Part-Time Job", ["Yes", "No"])

    with col3:
        history = st.selectbox(
            "Previous Mental Health History",
            ["Yes", "No"]
        )

    st.divider()
    st.subheader("📚 Academic & Lifestyle Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        sleep = st.number_input(
            "Sleep Hours per Day", 0.0, 24.0, 7.0
        )
        study = st.number_input(
            "Study Hours per Week", 0.0, 168.0, 20.0
        )
        gpa = st.number_input(
            "GPA", 0.0, 4.0, 3.0
        )

    with col2:
        social = st.number_input(
            "Social Media Hours per Day", 0.0, 24.0, 3.0
        )
        screen = st.number_input(
            "Screen Time Hours per Day", 0.0, 24.0, 5.0
        )
        activity = st.selectbox(
            "Physical Activity",
            ["Low", "Moderate", "High"]
        )

    with col3:
        financial = st.number_input(
            "Financial Stress (0-10)", 0.0, 10.0, 5.0
        )
        family = st.number_input(
            "Family Support (0-10)", 0.0, 10.0, 5.0
        )
        attendance = st.number_input(
            "Attendance Rate (%)", 0.0, 100.0, 80.0
        )

    diet = st.selectbox(
        "Diet Quality",
        ["Poor", "Average", "Good"]
    )

    submitted = st.form_submit_button(
        "🔍 Predict Mental Health Risk",
        use_container_width=True
    )


# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Year_of_Study": year,
        "Sleep_Hours": sleep,
        "Study_Hours_Per_Week": study,
        "GPA": gpa,
        "Social_Media_Hours": social,
        "Screen_Time_Hours": screen,
        "Physical_Activity": activity,
        "Financial_Stress": financial,
        "Family_Support": family,
        "Attendance_Rate": attendance,
        "Part_Time_Job": part_time,
        "Diet_Quality": diet,
        "Previous_Mental_Health_History": history
    }])

    try:
        prediction = model.predict(input_data)[0]

        labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        result = labels[int(prediction)]

        st.divider()
        st.subheader("Prediction Result")

        if result == "Low":
            st.success("Risk Level: LOW 🟢")

        elif result == "Medium":
            st.warning("Risk Level: MEDIUM 🟠")

        else:
            st.error("Risk Level: HIGH 🔴")

        st.write(f"Predicted class: **{result}**")

        st.caption(
            "This prediction is an experimental ML result, "
            "not a clinical diagnosis or substitute for professional assessment."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
