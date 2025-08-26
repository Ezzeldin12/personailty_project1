import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CSV_PATH = r"C:\Users\zuz12\OneDrive\Desktop\personality_prediction\personality_dataset.csv"

def main():
    df = pd.read_csv(CSV_PATH).copy()

    # Encode Yes/No
    bin_cols = []
    for col in df.columns:
        if df[col].dtype == object and set(df[col].dropna().unique()).issubset({"Yes", "No"}):
            bin_cols.append(col)
    for col in bin_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    target_col = "Personality"
    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col])

    for col in X.columns:
        if X[col].dtype != object:
            X[col] = X[col].fillna(X[col].median())

    non_num = [c for c in X.columns if X[c].dtype == object]
    if non_num:
        for c in non_num:
            X[c] = X[c].astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_train_s, y_train)

    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(clf, "model_svm.joblib")

    schema = {
        "feature_order": list(X.columns),
        "binary_columns_mapped_from_yes_no": bin_cols,
        "target": target_col,
        "classes_": sorted(list(y.unique())),
    }
    with open("schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print("✅ Done. Saved: scaler.joblib, model_svm.joblib, schema.json")

if __name__ == "__main__":
    main()
