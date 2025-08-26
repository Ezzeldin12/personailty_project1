# Personality Prediction Project — SVM + Streamlit

## 📌 Introduction
This project predicts whether a person is an **Introvert** or an **Extrovert** based on their social habits.  
The model is trained with **Support Vector Machine (SVM)** and deployed with **Streamlit**.

👉 Try it for yourself, your **brother**, your **friend**, or even your **kid** by entering their lifestyle habits.

---

## 📖 Project Description
- **Goal:** Predict personality type from lifestyle & social features.  
- **Dataset:** `personality_dataset.csv` (behavioral survey-like data).  
- **Model:** SVM (RBF kernel, probability=True).  
- **Deployment:** Streamlit app.  
- **Artifacts:** `model_svm.joblib`, `scaler.joblib`, `schema.json`.  

---

## 🔄 Project Workflow

### 1. Data Preparation
- Columns include:
  - `Time_spent_Alone`  
  - `Stage_fear`  
  - `Social_event_attendance`  
  - `Going_outside`  
  - `Drained_after_socializing`  
  - `Friends_circle_size`  
  - `Post_frequency`  
- Target: `Personality` ∈ {Introvert, Extrovert}  

**Handling Missing Data**  
- Checked percentage of missing values.  
- Since the percentage was **very small**, missing rows were **dropped** instead of filling with median/mean.  
- This avoids introducing artificial values and keeps dataset integrity.  

---

### 2. Training Pipeline (`train_svm.py`)
1. Load dataset.  
2. Drop missing rows (low percentage → safe).  
3. Encode binary columns (Yes/No → 1/0).  
4. Split into features (`X`) and target (`y`).  
5. Train/Test split (80/20, stratified).  
6. Standardize features using `StandardScaler` (fit only on train).  
7. Train SVM (RBF kernel, probability=True).  
8. Save artifacts (`scaler.joblib`, `model_svm.joblib`, `schema.json`).  

---

## 🛡️ Avoiding Data Leakage
- **Scaling:** Fitted scaler only on **training data**, then applied to test.  
- **Encoding:** Applied consistent mappings (not influenced by target).  
- **Schema file:** Ensures the app reuses training schema (feature order & mappings).  
- **Missing data:** Rows dropped **before splitting**, so no leakage of test distribution into training.

✅ This guarantees evaluation reflects **true generalization**.

---

## 3. Inference App (`app_streamlit.py`)
- Loads artifacts (`scaler`, `model`, `schema.json`).  
- User inputs numeric + Yes/No values.  
- Reindexes data to match training schema.  
- Applies saved scaler.  
- Outputs:
  - Predicted label  
  - Class probabilities  

---

## 4. Deployment
- Local run:
  ```bash
  streamlit run app_streamlit.py
