import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. LOAD DATASETS
# ==========================================

features = pd.read_csv("plant_features.csv")
mapping = pd.read_csv("plant_mapping_current.csv")

# Merge extracted features with actual labels
data = features.merge(
    mapping[["Image", "Disease", "Split"]],
    on=["Image", "Disease", "Split"],
    how="inner",
    validate="one_to_one"
)

print("Total matched images:", len(data))

# ==========================================
# 2. SELECT FEATURES AND LABELS
# ==========================================

feature_columns = [
    "Mean_R",
    "Mean_G",
    "Mean_B",
    "Std_R",
    "Std_G",
    "Std_B"
]

X_train = data[data["Split"] == "train"][feature_columns]
y_train = data[data["Split"] == "train"]["Disease"]

X_val = data[data["Split"] == "val"][feature_columns]
y_val = data[data["Split"] == "val"]["Disease"]

X_test = data[data["Split"] == "test"][feature_columns]
y_test = data[data["Split"] == "test"]["Disease"]

print("\nDataset distribution:")
print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))

# ==========================================
# 3. TRAIN RANDOM FOREST MODEL
# ==========================================

print("\nTraining Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 4. VALIDATION EVALUATION
# ==========================================

val_predictions = rf_model.predict(X_val)

val_accuracy = accuracy_score(y_val, val_predictions)

print("\nValidation Accuracy:", round(val_accuracy, 4))

# ==========================================
# 5. FINAL TRAINING WITH TRAIN + VAL
# ==========================================

X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

print("\nRetraining using training + validation data...")

final_model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_final, y_train_final)

print("Final model training completed!")

# ==========================================
# 6. TEST PREDICTION
# ==========================================

test_predictions = final_model.predict(X_test)

# ==========================================
# 7. CALCULATE EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(y_test, test_predictions)

precision_macro = precision_score(
    y_test, test_predictions, average="macro", zero_division=0
)

recall_macro = recall_score(
    y_test, test_predictions, average="macro", zero_division=0
)

f1_macro = f1_score(
    y_test, test_predictions, average="macro", zero_division=0
)

precision_weighted = precision_score(
    y_test, test_predictions, average="weighted", zero_division=0
)

recall_weighted = recall_score(
    y_test, test_predictions, average="weighted", zero_division=0
)

f1_weighted = f1_score(
    y_test, test_predictions, average="weighted", zero_division=0
)

# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"Accuracy:           {accuracy:.4f}")
print(f"Macro Precision:    {precision_macro:.4f}")
print(f"Macro Recall:       {recall_macro:.4f}")
print(f"Macro F1-score:     {f1_macro:.4f}")

print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall:    {recall_weighted:.4f}")
print(f"Weighted F1-score:  {f1_weighted:.4f}")

# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    test_predictions,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    "random_forest_classification_report.csv"
)

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

# ==========================================
# 10. SAVE CONFUSION MATRIX
# ==========================================

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

fig, ax = plt.subplots(figsize=(16, 13))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=True
)

plt.title("Random Forest - Plant Disease Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 11. SAVE METRICS
# ==========================================

metrics = {
    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-score",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1-score"
    ],

    "Score": [
        accuracy,
        precision_macro,
        recall_macro,
        f1_macro,
        precision_weighted,
        recall_weighted,
        f1_weighted
    ]
}

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    "random_forest_metrics.csv",
    index=False
)

# ==========================================
# 12. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    final_model,
    "random_forest_model.pkl"
)

# ==========================================
# 13. SAVE TEST PREDICTIONS
# ==========================================

predictions_df = pd.DataFrame({
    "Actual_Disease": y_test.values,
    "Predicted_Disease": test_predictions
})

predictions_df.to_csv(
    "random_forest_predictions.csv",
    index=False
)

# ==========================================
# 14. FINAL SUMMARY
# ==========================================

print("\n==========================================")
print("RANDOM FOREST EVALUATION COMPLETED!")
print("==========================================")

print("Test images evaluated:", len(X_test))
print("Accuracy:", round(accuracy, 4))
print("Macro F1-score:", round(f1_macro, 4))

print("\nGenerated files:")
print("1. random_forest_metrics.csv")
print("2. random_forest_classification_report.csv")
print("3. random_forest_confusion_matrix.png")
print("4. random_forest_model.pkl")
print("5. random_forest_predictions.csv")

