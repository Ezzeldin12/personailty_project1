# Personality Prediction Project — SVM + Streamlit

## 📌 Introduction
This project predicts whether a person is an **Introvert** or an **Extrovert** based on their lifestyle and social habits.  
The model is trained using a **Support Vector Machine (SVM)** with RBF kernel and deployed as an interactive **Streamlit app**.  

👉 You can try it for yourself, your **brother**, your **friend**, or even your **kid**.

---

## 📖 Project Description
- **Goal:** Predict personality type (Introvert/Extrovert) from behavioral features.  
- **Dataset:** `personality_dataset.csv`.  
- **Model:** Support Vector Machine (RBF kernel, probability=True).  
- **Deployment:** `app_streamlit.py` (Streamlit).  
- **Artifacts:** `model_svm.joblib`, `scaler.joblib`, `schema.json`.  

---

## 🔄 Project Workflow

### 1. Data Preparation
- Dropped missing rows (since percentage was very small).  
- Applied **Label Encoding** for categorical columns (`Stage_fear`, `Drained_after_socializing`, `Personality`).  

### 2. Train/Test Split
- 80% train / 20% test with stratification.  
- Performed **before scaling** and **before correlation analysis**.  

### 3. Correlation Analysis
- Conducted only on the **training set** after splitting.  
- Ensures no test data influences feature exploration.  

### 4. Scaling
- Applied **StandardScaler**:  
  - Fit on training set only.  
  - Applied transformation to both train and test.  

### 5. Model Training
- Used `SVC(kernel="rbf", probability=True)`.  
- Trained on scaled training data.  
- Saved artifacts: `scaler.joblib`, `model_svm.joblib`, `schema.json`.  

### 6. Inference App
- Streamlit form for manual input.  
- Data reindexed to match training schema.  
- Applies saved scaler → model prediction → probabilities shown.  

---

## 🛡️ Data Leakage: How I Avoided It
Data leakage happens when **information from the test set** sneaks into training, giving artificially high accuracy.  
I carefully designed the pipeline to prevent this:
  
1. **Performed Train/Test split before scaling or correlation**  
   → test data never influences preprocessing or analysis.  

2. **Scaler fitted only on training set**  
   → applied later to test set, keeping it isolated.  

3. **Correlation analysis done on training data only**  
   → prevents knowledge of test distribution from leaking into training.  

4. **Label Encoding applied consistently**  
   → same encoder mapping used across train/test without refitting on test.  

✅ With these steps, the model evaluation reflects **true generalization performance**.  

---

## ✅ Summary
- Clean dataset (dropped few missing rows).  
- Label Encoding for categorical variables.  
- Train/Test split → correlation → scaling → training.  
- Independent **Data Leakage section** ensures clarity on prevention steps.  
- Deployed via Streamlit for interactive predictions.  
