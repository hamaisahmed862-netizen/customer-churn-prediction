import joblib
import pandas as pd
from tensorflow.keras.models import load_model


# ==========================================
# 1. Load Model and Preprocessor
# ==========================================

model = load_model("model.keras")

preprocessor = joblib.load(
    "preprocessor.joblib"
)


# ==========================================
# 2. Create New Customer
# ==========================================

customer = pd.DataFrame([{

    "Gender": "Male",

    "Senior Citizen": "No",

    "Partner": "No",

    "Dependents": "No",

    "Tenure Months": 5,

    "Phone Service": "Yes",

    "Multiple Lines": "No",

    "Internet Service": "Fiber optic",

    "Online Security": "No",

    "Online Backup": "No",

    "Device Protection": "No",

    "Tech Support": "No",

    "Streaming TV": "Yes",

    "Streaming Movies": "Yes",

    "Contract": "Month-to-month",

    "Paperless Billing": "Yes",

    "Payment Method": "Electronic check",

    "Monthly Charges": 90.0,

    "Total Charges": 450.0,

    "CLTV": 5000

}])


# ==========================================
# 3. Preprocess Customer
# ==========================================

customer_processed = preprocessor.transform(
    customer
)


# ==========================================
# 4. Predict
# ==========================================

probability = model.predict(
    customer_processed,
    verbose=0
)[0][0]


prediction = int(
    probability >= 0.5
)


# ==========================================
# 5. Display Result
# ==========================================

print("\nCustomer Churn Prediction")
print("-------------------------")

print(
    f"Churn Probability: {probability:.2%}"
)

if prediction == 1:

    print("Prediction: Churn")

else:

    print("Prediction: No Churn")

