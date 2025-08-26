# Personality Prediction Project — SVM + Streamlit

## 📌 Introduction
This project is a **machine learning pipeline** that predicts whether a person is an **Introvert** or an **Extrovert** based on their social habits.  
The model is trained using **Support Vector Machine (SVM)** and deployed with an interactive **Streamlit app**.  

👉 The idea is simple: you enter behaviors (e.g., hours spent alone, friends circle size, posting frequency), and the app tells you the predicted personality.  
You can try it for yourself, your **brother**, your **friend**, or even your **kid**.

---

## 📖 Project Description
- **Goal:** Predict personality type (Introvert/Extrovert) from lifestyle & social features.  
- **Dataset:** Behavioral data stored in `personality_dataset.csv`.  
- **Model:** SVM (RBF kernel, probability=True).  
- **Deployment:** Interactive web app via Streamlit (`app_streamlit.py`).  
- **Artifacts:** Saved model (`model_svm.joblib`), scaler (`scaler.joblib`), schema (`schema.json`).  

---

## 🔄 Project Workflow (Step by Step)

### 1. Data Preparation
- File: `personality_dataset.csv`  
- Features:
  - `Time_spent_Alone` (numeric, hours/day)  
  - `Stage_fear` (Yes/No → mapped to 1/0)  
  - `Social_event_attendance` (numeric, per month)  
  - `Going_outside` (numeric, per week)  
  - `Drained_after_socializing` (Yes/No → mapped to 1/0)  
  - `Friends_circle_size` (numeric, number of close friends)  
  - `Post_frequency` (numeric, posts/week)  
- Target: `Personality` ∈ {Introvert, Extrovert}  

### 2. Training Pipeline (`train_svm.py`)
1. **Load dataset** with pandas.  
2. **Binary encode** Yes/No → 1/0.  
3. **Split** features (`X`) and target (`y`).  
4. **Handle missing values** → numeric columns filled with median.  
5. **Encode categorical** columns (if any left).  
6. **Split train/test** (80/20, stratified).  
7. **Scale features** using `StandardScaler`.  
8. **Train SVM** with RBF kernel and `probability=True`.  
9. **Save artifacts**:  
   - `scaler.joblib`  
   - `model_svm.joblib`  
   - `schema.json` (feature order, binary cols, target classes)  

### 3. Artifacts
- **scaler.joblib** → ensures same scaling at inference.  
- **model_svm.joblib** → trained SVM classifier.  
- **schema.json** → defines feature order + class labels.  

### 4. Inference App (`app_streamlit.py`)
- Loads `scaler`, `model`, and `schema.json`.  
- User enters values:
  - Numbers → `st.number_input`  
  - Yes/No → `st.selectbox`  
- Data is reindexed to match training schema.  
- Same scaler applied.  
- Model predicts:
  - **Prediction** (Introvert/Extrovert)  
  - **Class probabilities** if available.  

### 5. Deployment
- **Local run:**  
  ```bash
  streamlit run app_streamlit.py
