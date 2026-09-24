import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


# ==========================================
# 1. Page Configuration
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# 2. Load Model and Preprocessor
# ==========================================

@st.cache_resource
def load_ml_components():

    model = load_model("model.keras")

    preprocessor = joblib.load(
        "preprocessor.joblib"
    )

    return model, preprocessor


model, preprocessor = load_ml_components()


# ==========================================
# 3. Title
# ==========================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter the customer's information below to predict "
    "their probability of churn."
)


# ==========================================
# 4. Customer Information
# ==========================================

st.header("Customer Information")

col1, col2 = st.columns(2)


# ==========================================
# Left Column
# ==========================================

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=100,
        value=5
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


# ==========================================
# Right Column
# ==========================================

with col2:

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=90.0,
        step=0.01
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=450.0,
        step=0.01
    )

    cltv = st.number_input(
        "CLTV",
        min_value=0,
        value=5000
    )


# ==========================================
# 5. Create Customer DataFrame
# ==========================================

customer = pd.DataFrame([{

    "Gender": gender,

    "Senior Citizen": senior_citizen,

    "Partner": partner,

    "Dependents": dependents,

    "Tenure Months": tenure,

    "Phone Service": phone_service,

    "Multiple Lines": multiple_lines,

    "Internet Service": internet_service,

    "Online Security": online_security,

    "Online Backup": online_backup,

    "Device Protection": device_protection,

    "Tech Support": tech_support,

    "Streaming TV": streaming_tv,

    "Streaming Movies": streaming_movies,

    "Contract": contract,

    "Paperless Billing": paperless_billing,

    "Payment Method": payment_method,

    "Monthly Charges": monthly_charges,

    "Total Charges": total_charges,

    "CLTV": cltv

}])


# ==========================================
# 6. Prediction Button
# ==========================================

if st.button(
    "Predict Churn",
    type="primary"
):

    # --------------------------------------
    # Preprocess customer data
    # --------------------------------------

    customer_processed = preprocessor.transform(
        customer
    )


    # --------------------------------------
    # Predict churn probability
    # --------------------------------------

    probability = model.predict(
        customer_processed,
        verbose=0
    )[0][0]


    # --------------------------------------
    # Convert probability to prediction
    # --------------------------------------

    prediction = int(
        probability >= 0.5
    )


    # ======================================
    # 7. Display Result
    # ======================================

    st.divider()

    st.header("Prediction Result")

    col1, col2 = st.columns(2)


    # --------------------------------------
    # Churn Probability
    # --------------------------------------

    with col1:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


    # --------------------------------------
    # Prediction
    # --------------------------------------

    with col2:

        if prediction == 1:

            st.error(
                "⚠️ Prediction: Churn"
            )

        else:

            st.success(
                "✅ Prediction: No Churn"
            )


