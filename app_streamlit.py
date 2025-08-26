import streamlit as st
import pandas as pd
import json, joblib

st.set_page_config(page_title="SVM Personality (Streamlit)", page_icon="🤖", layout="centered")
st.title("🤖 personality prediction by Ezzeldin Shady ")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("model_svm.joblib")
    with open("schema.json") as f:
        schema = json.load(f)
    return scaler, model, schema

scaler, model, schema = load_artifacts()
FEATURES = schema["feature_order"]
CLASSES = schema["classes_"]

with st.form("form"):
    st.subheader("Enter the features:")

    Time_spent_Alone = st.number_input("Time Spent Alone (hours/day)", min_value=0, max_value=24, value=5)
    st.caption("How many hours per day do you usually spend alone?")

    Stage_fear = st.selectbox("Stage Fear", ["No", "Yes"], index=0)
    st.caption("Do you feel nervous when speaking or performing on stage?")

    Social_event_attendance = st.number_input("Social Event Attendance (per month)", min_value=0, value=5)
    st.caption("How many social events do you attend in a month?")

    Going_outside = st.number_input("Going Outside (per week)", min_value=0, value=5)
    st.caption("How many times per week do you usually go outside?")

    Drained_after_socializing = st.selectbox("Drained After Socializing", ["No","Yes"], index=0)
    st.caption("Do you feel drained or exhausted after social interactions?")

    Friends_circle_size = st.number_input("Friends Circle Size", min_value=0, value=10)
    st.caption("Approximately how many close friends are in your circle?")

    Post_frequency = st.number_input("Post Frequency (per week)", min_value=0, value=5)
    st.caption("How often do you post on social media per week?")

    submitted = st.form_submit_button("Predict")

if submitted:
    data = {
        "Time_spent_Alone": Time_spent_Alone,
        "Stage_fear": 1 if Stage_fear == "Yes" else 0,
        "Social_event_attendance": Social_event_attendance,
        "Going_outside": Going_outside,
        "Drained_after_socializing": 1 if Drained_after_socializing == "Yes" else 0,
        "Friends_circle_size": Friends_circle_size,
        "Post_frequency": Post_frequency,
    }

    df = pd.DataFrame([data]).reindex(columns=FEATURES, fill_value=0)
    X = scaler.transform(df.values)

    pred = model.predict(X)[0]
    st.success(f"✅ Prediction: **{pred}**")

    try:
        proba = model.predict_proba(X)[0]
        st.write("📊 Probabilities:")
        for cls, p in sorted(zip(CLASSES, proba), key=lambda kv: kv[1], reverse=True):
            st.write(f"- {cls}: {p:.3f}")
    except Exception:
        st.info("⚠️ Model does not expose predict_proba.")
