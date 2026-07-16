import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/customer_purchase_data.csv")

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    ),
}

results = []
trained_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
        }
    )

    trained_models[model_name] = model

results_df = pd.DataFrame(results).sort_values(
    by="Accuracy",
    ascending=False,
)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False,
)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "outputs/best_purchase_model.pkl",
)

best_predictions = best_model.predict(X_test)

report = classification_report(
    y_test,
    best_predictions,
)

with open(
    "outputs/classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report)

confusion = confusion_matrix(
    y_test,
    best_predictions,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
)

display.plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(
    "outputs/confusion_matrix.png",
)
plt.close()

if best_model_name == "Random Forest":
    feature_importance = best_model.feature_importances_

elif best_model_name == "Decision Tree":
    feature_importance = best_model.feature_importances_

else:
    logistic_model = best_model.named_steps["model"]
    feature_importance = abs(
        logistic_model.coef_[0]
    )

importance_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": feature_importance,
    }
).sort_values(
    by="Importance",
    ascending=False,
)

importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False,
)

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"],
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
)

plt.close()

print("Model training completed.")
print(results_df)
print(f"Best model: {best_model_name}")