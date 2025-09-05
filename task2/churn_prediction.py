# spotify_churn_model.py

# ===== Step 1: Import Libraries =====
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc
)
import seaborn as sns

# ===== Step 2: Load Dataset =====
df = pd.read_excel("Spotify_data.xlsx")

# ===== Step 3: Feature Encoding =====
target_col = 'premium_sub_willingness'
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = encoder.fit_transform(df[col])

# ===== Step 4: Split Data =====
X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== Step 5: Train Models =====
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

results = {}
plt.figure(figsize=(12, 5))

for i, (name, model) in enumerate(models.items(), start=1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # For ROC

    # Store metrics
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Report": classification_report(y_test, y_pred)
    }

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(2, len(models), i)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # === ROC Curve ===
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.subplot(2, len(models), i + len(models))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

plt.tight_layout()
plt.show()

# ===== Step 6: Feature Importance (Random Forest) =====
importances = models['Random Forest'].feature_importances_
features = X.columns
plt.figure(figsize=(8, 6))
plt.barh(features, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest')
plt.show()

# ===== Step 7: Save Data with Predictions =====
age_mapping = {0: "18-24", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55+"}
gender_mapping = {0: "Male", 1: "Female", 2: "Other"}

df['Predicted_Premium_Willingness'] = models['Random Forest'].predict(X)
if 'Age' in df.columns:
    df['Age'] = df['Age'].map(age_mapping)
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map(gender_mapping)

df.to_csv("spotify_predictions.csv", index=False, encoding='utf-8')
print("Saved predictions to spotify_predictions.csv with readable Age & Gender")
