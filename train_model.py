import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

y = df["Churn Value"]

drop_cols = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason",
    "Churn Value"
]

existing_drop_cols = [col for col in drop_cols if col in df.columns]
X = df.drop(columns=existing_drop_cols)

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

models = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_model_name = None
best_model_pipeline = None
best_auc = 0

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        auc = 0

    acc = accuracy_score(y_test, y_pred)

    print(f"\nModel: {name}")
    print("Accuracy:", round(acc, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model_pipeline = pipeline

joblib.dump(best_model_pipeline, MODEL_DIR / "best_customer_leaving_model.joblib")

top_feature_importance = []

try:
    preprocessor_fitted = best_model_pipeline.named_steps["preprocessor"]
    model_fitted = best_model_pipeline.named_steps["model"]

    feature_names = preprocessor_fitted.get_feature_names_out()

    if hasattr(model_fitted, "feature_importances_"):
        importances = model_fitted.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        top_feature_importance = fi_df.head(20).to_dict(orient="records")
except Exception as e:
    print("Could not extract feature importances:", e)

metadata = {
    "best_model_name": best_model_name,
    "roc_auc": float(best_auc),
    "features": X.columns.tolist(),
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "target_name": "Churn Value",
    "class_labels": {"0": "Stay", "1": "Leave"},
    "risk_thresholds": {
        "low": 0.35,
        "medium": 0.70
    },
    "top_feature_importance": top_feature_importance
}

joblib.dump(metadata, MODEL_DIR / "model_metadata.joblib")

print(f"\nBest model saved: {best_model_name}")
print(f"Best ROC-AUC: {best_auc:.4f}")