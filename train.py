import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.preprocessing import preprocessor


# ==========================================
# 1. Load Data
# ==========================================

df = pd.read_excel("data/telco_churn.xlsx")


# Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)


# ==========================================
# 2. Select Features and Target
# ==========================================

features = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "CLTV"
]

X = df[features]
y = df["Churn Value"]


# ==========================================
# 3. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Preprocess Data
# ==========================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ==========================================
# 5. Build Neural Network
# ==========================================

model = Sequential([
    Input(shape=(X_train_processed.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])


# ==========================================
# 6. Compile Model
# ==========================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 7. Early Stopping
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ==========================================
# 8. Calculate Class Weights
# ==========================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)


# ==========================================
# 9. Train Model
# ==========================================

history = model.fit(
    X_train_processed,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.20,
    callbacks=[early_stopping],
    class_weight=class_weights
)


# ==========================================
# 10. Evaluate Model
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test_processed,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ==========================================
# 11. Generate Predictions
# ==========================================

y_probability = model.predict(
    X_test_processed,
    verbose=0
).flatten()

y_prediction = (y_probability >= 0.5).astype(int)


# ==========================================
# 12. Classification Report
# ==========================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_prediction
    )
)


# ==========================================
# 13. Confusion Matrix
# ==========================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_prediction
    )
)


# ==========================================
# 14. ROC-AUC
# ==========================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC:", roc_auc)


# ==========================================
# 15. Threshold Evaluation
# ==========================================

thresholds_to_test = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7
]

print("\nThreshold Comparison:")

for threshold in thresholds_to_test:

    predictions = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

# ==========================================
# 16. Save Trained Model
# ==========================================

model.save("model.keras")

print("\nModel saved successfully as model.keras")

joblib.dump(
    preprocessor,
    "preprocessor.joblib"
)

print("Preprocessor saved successfully as preprocessor.joblib")