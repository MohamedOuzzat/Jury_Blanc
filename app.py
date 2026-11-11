"""Student Mental Health Risk - educational Streamlit prototype.

Not a diagnostic, screening, disciplinary or selection tool.
"""
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Paths (override with environment variables if needed)
# ----------------------------------------------------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/opt/airflow/Models/Best/best_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(MODEL_PATH.with_name("metrics.json"))))

ORDER = ["Low", "Medium", "High"]
LABELS = {0: "Low", 1: "Medium", 2: "High"}
COLORS = {"Low": "#3f9d77", "Medium": "#d9962b", "High": "#d8574f"}
GUIDANCE = {
    "Low": "These answers look similar to profiles the model groups as lower risk. "
           "Regular sleep, movement and time with other people all help keep it that way.",
    "Medium": "These answers sit between the two ends. Small changes to routine can matter, "
              "and the student wellbeing service is a good place to ask questions.",
    "High": "These answers look similar to profiles the model groups as higher risk. "
            "Talking with a counselor or someone you trust is a good next step.",
}
FEATURES = [
    "Age", "Gender", "Year_of_Study", "Sleep_Hours", "Study_Hours_Per_Week", "GPA",
    "Social_Media_Hours", "Screen_Time_Hours", "Physical_Activity", "Financial_Stress",
    "Family_Support", "Attendance_Rate", "Part_Time_Job", "Diet_Quality",
    "Previous_Mental_Health_History",
]

PRESETS = {
    "Steady routine": dict(age=21, gender="Female", year=2, sleep=7.8, study=20.0, gpa=3.3,
                           social=2.5, screen=6.0, activity="High", financial=3.0, family=8.5,
                           attendance=94.0, job="No", diet="Good", history="No"),
    "Somewhere in between": dict(age=21, gender="Male", year=3, sleep=6.4, study=22.0, gpa=2.8,
                                 social=5.0, screen=8.0, activity="Moderate", financial=5.5,
                                 family=5.5, attendance=82.0, job="No", diet="Average", history="No"),
    "Heavy load": dict(age=22, gender="Male", year=4, sleep=4.8, study=30.0, gpa=2.3,
                       social=7.5, screen=11.0, activity="Low", financial=8.5, family=3.5,
                       attendance=70.0, job="Yes", diet="Poor", history="Yes"),
}
DEFAULTS = PRESETS["Somewhere in between"]

st.set_page_config(page_title="Student Mental Health Risk", page_icon="🧭", layout="wide")

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1180px;}
.masthead {border-bottom: 1px solid rgba(128,128,128,.28); padding-bottom: 1.1rem; margin-bottom: 1.2rem;}
.masthead h1 {font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; font-weight: 600;
  font-size: 2.5rem; letter-spacing: -0.02em; margin: 0 0 .3rem 0; padding: 0;}
.masthead p {margin: 0; opacity: .72; max-width: 60ch; line-height: 1.5;}
.verdict {font-family: Georgia, "Iowan Old Style", serif; font-size: 2.1rem; font-weight: 600;
  letter-spacing: -0.015em; line-height: 1.15; margin: .2rem 0 1rem 0;}
.verdict small {display: block; font-family: system-ui, sans-serif; font-size: .9rem; font-weight: 400;
  opacity: .65; letter-spacing: 0; margin-bottom: .25rem;}
.spec {margin: .4rem 0 1.5rem 0;}
.spec-track {position: relative; height: 12px; border-radius: 999px;
  background: linear-gradient(90deg, #3f9d77 0%, #d9962b 50%, #d8574f 100%);}
.spec-pin {position: absolute; top: 50%; width: 26px; height: 26px; border-radius: 50%;
  transform: translate(-50%, -50%); background: #fff; border: 5px solid var(--pin);
  box-shadow: 0 2px 8px rgba(0,0,0,.35);}
.spec-labels {display: flex; justify-content: space-between; margin-top: .55rem; font-size: .85rem; opacity: .7;}
.prow {display: grid; grid-template-columns: 70px 1fr 44px; align-items: center; gap: .7rem;
  font-size: .92rem; margin: .35rem 0;}
.prow .bar {height: 6px; border-radius: 6px; background: rgba(128,128,128,.22); overflow: hidden;}
.prow .bar > span {display: block; height: 100%; border-radius: 6px;}
.prow .pct {text-align: right; opacity: .75; font-variant-numeric: tabular-nums;}
.guidance {border-left: 3px solid var(--pin); padding: .15rem 0 .15rem .9rem; margin: 1.3rem 0 .8rem 0;
  line-height: 1.55;}
.empty {border: 1px dashed rgba(128,128,128,.45); border-radius: 10px; padding: 2rem 1.4rem;
  opacity: .8; line-height: 1.55;}
.note {font-size: .82rem; opacity: .6; line-height: 1.45;}
div.stButton > button, div.stDownloadButton > button {width: 100%; border-radius: 8px; font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def to_label(value) -> str:
    try:
        return LABELS[int(value)]
    except (ValueError, TypeError, KeyError):
        return str(value)


def run_model(model, frame: pd.DataFrame):
    """Return (labels, probabilities or None)."""
    labels = [to_label(v) for v in model.predict(frame)]
    proba = None
    try:
        raw = model.predict_proba(frame)
        proba = pd.DataFrame(raw, columns=[to_label(c) for c in model.classes_])[ORDER]
    except Exception:
        pass
    return labels, proba


# ----------------------------------------------------------------------------
# Form state
# ----------------------------------------------------------------------------
for key, value in DEFAULTS.items():
    st.session_state.setdefault(f"f_{key}", value)


def apply_preset():
    chosen = st.session_state.get("preset")
    if chosen in PRESETS:
        for key, value in PRESETS[chosen].items():
            st.session_state[f"f_{key}"] = value


def read_inputs() -> dict:
    s = st.session_state
    return {
        "Age": s.f_age, "Gender": s.f_gender, "Year_of_Study": s.f_year,
        "Sleep_Hours": s.f_sleep, "Study_Hours_Per_Week": s.f_study, "GPA": s.f_gpa,
        "Social_Media_Hours": s.f_social, "Screen_Time_Hours": s.f_screen,
        "Physical_Activity": s.f_activity, "Financial_Stress": s.f_financial,
        "Family_Support": s.f_family, "Attendance_Rate": s.f_attendance,
        "Part_Time_Job": s.f_job, "Diet_Quality": s.f_diet,
        "Previous_Mental_Health_History": s.f_history,
    }


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def render_result(level: str, proba: dict | None):
    color = COLORS[level]
    if proba:
        position = proba["Medium"] * 50 + proba["High"] * 100
        confidence = f"{proba[level] * 100:.0f}% of the model's weight"
    else:
        position = {"Low": 0, "Medium": 50, "High": 100}[level]
        confidence = ""
    position = min(max(position, 4), 96)

    html = [
        f'<div style="--pin:{color}">',
        f'<div class="verdict"><small>Estimated risk level</small>{level}</div>',
        '<div class="spec"><div class="spec-track">',
        f'<div class="spec-pin" style="left:{position:.1f}%"></div></div>',
        '<div class="spec-labels"><span>Low</span><span>Medium</span><span>High</span></div></div>',
    ]
    if proba:
        for name in ORDER:
            pct = proba[name] * 100
            html.append(
                f'<div class="prow"><span>{name}</span>'
                f'<div class="bar"><span style="width:{pct:.1f}%;background:{COLORS[name]}"></span></div>'
                f'<span class="pct">{pct:.0f}%</span></div>'
            )
    html.append(f'<div class="guidance">{GUIDANCE[level]}</div>')
    if confidence:
        html.append(f'<div class="note">The chosen level carries {confidence}.</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def plot_confusion(cm, classes, normalize: bool):
    cm = np.array(cm, dtype=float)
    data = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, color="#888")
    ax.set_yticks(range(len(classes)), classes, color="#888")
    ax.set_xlabel("Predicted", color="#888")
    ax.set_ylabel("Actual", color="#888")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:.0%}" if normalize else f"{int(data[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if data[i, j] > data.max() / 2 else "#1f2937")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Masthead + sidebar
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="masthead"><h1>Student Mental Health Risk</h1>'
    "<p>An educational prototype. Explore how habits and context relate to an estimated "
    "risk level, and see how well the model performs.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Model")
    model_ready = MODEL_PATH.exists()
    if model_ready:
        st.success(f"Loaded: {MODEL_PATH.name}")
    else:
        st.error("Model file not found")
        st.caption(str(MODEL_PATH))
    st.divider()
    st.subheader("About")
    st.markdown(
        '<div class="note">This tool is for learning only. It is not a diagnosis, a clinical '
        "screening, a disciplinary tool or a way to select students. If you or someone you know "
        "is struggling, contact a health professional or your local emergency service.</div>",
        unsafe_allow_html=True,
    )

if not model_ready:
    st.error(f"No model at {MODEL_PATH}. Run the Airflow DAG first, or set the MODEL_PATH variable.")
    st.stop()

model = load_model(str(MODEL_PATH))

tab_estimate, tab_batch, tab_perf = st.tabs(["Estimate", "Batch file", "Model performance"])

# ----------------------------------------------------------------------------
# Tab 1: single estimate
# ----------------------------------------------------------------------------
with tab_estimate:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.selectbox("Start from an example", ["Custom"] + list(PRESETS), key="preset",
                     on_change=apply_preset)

        st.markdown("**About the student**")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Age", 15, 60, key="f_age")
        c2.selectbox("Gender", ["Female", "Male", "Non-binary"], key="f_gender")
        c3.selectbox("Year of study", [1, 2, 3, 4, 5], key="f_year")
        c1, c2 = st.columns(2)
        c1.selectbox("Part-time job", ["No", "Yes"], key="f_job")
        c2.selectbox("Previous mental health history", ["No", "Yes"], key="f_history")

        st.markdown("**Studies**")
        c1, c2, c3 = st.columns(3)
        c1.number_input("GPA", 0.0, 4.0, step=0.1, key="f_gpa")
        c2.number_input("Study hours per week", 0.0, 100.0, step=1.0, key="f_study")
        c3.number_input("Attendance rate (%)", 0.0, 100.0, step=1.0, key="f_attendance")

        st.markdown("**Daily life**")
        c1, c2 = st.columns(2)
        c1.slider("Sleep hours per night", 0.0, 14.0, step=0.1, key="f_sleep")
        c2.slider("Social media hours per day", 0.0, 16.0, step=0.1, key="f_social")
        c1.slider("Screen time hours per day", 0.0, 20.0, step=0.1, key="f_screen")
        c2.selectbox("Physical activity", ["Low", "Moderate", "High"], key="f_activity")
        c1.selectbox("Diet quality", ["Poor", "Average", "Good"], key="f_diet")

        st.markdown("**Support and pressure**")
        c1, c2 = st.columns(2)
        c1.slider("Financial stress (0 to 10)", 0.0, 10.0, step=0.1, key="f_financial")
        c2.slider("Family support (0 to 10)", 0.0, 10.0, step=0.1, key="f_family")

        if st.session_state.f_social > st.session_state.f_screen:
            st.warning("Social media time is higher than total screen time. Check these two values.")

        go = st.button("Estimate risk level", type="primary")

    inputs = read_inputs()
    if go:
        labels, proba = run_model(model, pd.DataFrame([inputs]))
        st.session_state.last = dict(
            level=labels[0],
            proba=None if proba is None else proba.iloc[0].to_dict(),
            snapshot=inputs,
        )

    with right:
        last = st.session_state.get("last")
        if last is None:
            st.markdown(
                '<div class="empty">Fill in the form, or pick an example, then choose '
                "<b>Estimate risk level</b>. The result appears here with the probability the "
                "model gives to each level.</div>",
                unsafe_allow_html=True,
            )
        else:
            render_result(last["level"], last["proba"])
            if last["snapshot"] != inputs:
                st.caption("The answers changed since this estimate. Estimate again to update it.")

# ----------------------------------------------------------------------------
# Tab 2: batch file
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown("Upload a CSV with one student per row to test the model on new data.")
    template = pd.DataFrame([{
        "Age": 21, "Gender": "Female", "Year_of_Study": 2, "Sleep_Hours": 7.0,
        "Study_Hours_Per_Week": 20.0, "GPA": 3.0, "Social_Media_Hours": 3.0,
        "Screen_Time_Hours": 6.0, "Physical_Activity": "Moderate", "Financial_Stress": 5.0,
        "Family_Support": 6.0, "Attendance_Rate": 85.0, "Part_Time_Job": "No",
        "Diet_Quality": "Average", "Previous_Mental_Health_History": "No",
    }])
    st.download_button("Download a CSV template", template.to_csv(index=False).encode("utf-8"),
                       "template.csv", "text/csv")

    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            labels, proba = run_model(model, data[FEATURES])
            out = data.copy()
            out["Predicted_Risk"] = labels
            if proba is not None:
                for name in ORDER:
                    out[f"P_{name}"] = proba[name].round(3).to_numpy()

            counts = pd.Series(labels).value_counts().reindex(ORDER, fill_value=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Students", len(out))
            for col, name in zip((m2, m3, m4), ORDER):
                col.metric(name, int(counts[name]))

            st.bar_chart(counts)
            try:
                shades = {k: f"background-color: {v}33" for k, v in COLORS.items()}
                st.dataframe(out.style.map(lambda v: shades.get(v, ""), subset=["Predicted_Risk"]))
            except Exception:
                st.dataframe(out)
            st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

# ----------------------------------------------------------------------------
# Tab 3: performance
# ----------------------------------------------------------------------------
with tab_perf:
    if not METRICS_PATH.exists():
        st.info(
            f"No metrics file at {METRICS_PATH}. Save one from your evaluation step to see "
            "the model's scores here."
        )
    else:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if metrics.get("model_name"):
            st.markdown(f"Model: **{metrics['model_name']}** (scores on the held-out test set)")

        shown = [("Accuracy", "accuracy"), ("Precision", "precision_macro"),
                 ("Recall", "recall_macro"), ("F1-score", "f1_macro"), ("ROC-AUC", "roc_auc")]
        cols = st.columns(len(shown))
        for col, (title, key) in zip(cols, shown):
            if key in metrics:
                col.metric(title, f"{metrics[key]:.3f}")

        st.divider()
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if "confusion_matrix" in metrics:
                st.markdown("**Confusion matrix**")
                norm = st.toggle("Show percentages per actual class")
                classes = metrics.get("classes", ORDER)
                st.pyplot(plot_confusion(metrics["confusion_matrix"], classes, norm))
        with c2:
            if "per_class" in metrics:
                st.markdown("**Scores per class**")
                st.dataframe(pd.DataFrame(metrics["per_class"]).T.round(3))
                st.caption("The Medium class is usually the hardest to separate from the other two.")
            if "models_comparison" in metrics:
                st.markdown("**Model comparison**")
                comp = pd.DataFrame(metrics["models_comparison"]).set_index("model")
                st.bar_chart(comp)