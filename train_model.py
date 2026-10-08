# ============================================================
# DIABETES PREDICTION PROJECT
# STEP 1 - TRAIN ALL ML MODELS
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv("diabetes.csv")

print("Dataset Shape:", df.shape)
print("\nDataset:")
print(df.head())


# -----------------------------
# 3. Separate Features & Target
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# 5. Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Save scaler
os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")


# -----------------------------
# 6. Create Models
# -----------------------------
models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),

    "Naive Bayes":
        GaussianNB(),

    "AdaBoost":
        AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric="logloss"
        )
}


# -----------------------------
# 7. Train Models
# -----------------------------
results = []

roc_data = {}


for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Models that need scaled data
    if name in ["Logistic Regression", "KNN", "SVM", "Naive Bayes"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Tree-based models
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]


    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)


    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))


    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    # -----------------------------
    # Classification Report
    # -----------------------------
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    # -----------------------------
    # Save Results
    # -----------------------------
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })


    # -----------------------------
    # ROC Data
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_data[name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc
    }


    # -----------------------------
    # Save Model
    # -----------------------------
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        "models/" + filename
    )


# -----------------------------
# 8. Results Table
# -----------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


# Save results
results_df.to_csv(
    "models/model_results.csv",
    index=False
)


# -----------------------------
# 9. Find Best Model
# -----------------------------
best_model_name = results_df.iloc[0]["Model"]
best_accuracy = results_df.iloc[0]["Accuracy"]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy, 4))


# Save best model name
with open("models/best_model.txt", "w") as file:
    file.write(best_model_name)


# -----------------------------
# 10. ROC Curve Comparison
# -----------------------------
plt.figure(figsize=(10, 7))

for name, data in roc_data.items():

    plt.plot(
        data["fpr"],
        data["tpr"],
        label=f"{name} (AUC = {data['auc']:.2f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "models/roc_comparison.png"
)

plt.show()


# -----------------------------
# 11. Accuracy Comparison
# -----------------------------
plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylabel("Accuracy")
plt.xlabel("Model")

plt.title("Model Accuracy Comparison")

plt.tight_layout()

plt.savefig(
    "models/accuracy_comparison.png"
)

plt.show()


print("\n===================================")
print("STEP 1 COMPLETED SUCCESSFULLY!")
print("===================================")
