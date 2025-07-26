import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰")

# Suppress the scikit-learn version mismatch warning for a cleaner interface
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")

# --- Load the Model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model from the joblib file."""
    try:
        model = joblib.load('loan_approval_model.joblib')
        return model
    except FileNotFoundError:
        # We handle the error message display within the main app body
        return None

model = load_model()

# --- Streamlit App Interface ---
st.title("💰 Loan Approval Prediction App")
st.write("""
This app predicts whether a loan application will be approved or rejected.
Please fill in the applicant's details in the sidebar to get a prediction.
""")

st.sidebar.header("Applicant Information")

def get_user_input():
    """Gets user input from the sidebar and returns it as a DataFrame."""
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    married = st.sidebar.selectbox("Married", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Dependents", ("0", "1", "2", "3+"))
    education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
    credit_history = st.sidebar.selectbox("Credit History Available", (1.0, 0.0), format_func=lambda x: 'Yes' if x == 1.0 else 'No')
    property_area = st.sidebar.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=1500)
    loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=10, value=150)
    loan_amount_term = st.sidebar.slider("Loan Amount Term (in months)", 36, 480, 360)
    
    # --- Pre-process User Input ---
    gender_code = 1 if gender == "Male" else 0
    married_code = 1 if married == "Yes" else 0
    education_code = 0 if education == "Graduate" else 1
    self_employed_code = 1 if self_employed == "Yes" else 0
    property_area_code = 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0)
    dependents_code = 3 if dependents == '3+' else int(dependents)
    applicant_income_log = np.log(applicant_income + 1)
    coapplicant_income_log = np.log(coapplicant_income + 1)
    loan_amount_log = np.log(loan_amount + 1)

    input_data = {
        'Gender': [gender_code], 'Married': [married_code], 'Dependents': [dependents_code],
        'Education': [education_code], 'Self_Employed': [self_employed_code],
        'Loan_Amount_Term': [loan_amount_term], 'Credit_History': [credit_history],
        'Property_Area': [property_area_code], 'ApplicantIncome_log': [applicant_income_log],
        'CoapplicantIncome_log': [coapplicant_income_log], 'LoanAmount_log': [loan_amount_log]
    }
    return pd.DataFrame(input_data)

user_input_df = get_user_input()

# --- Prediction and Display ---
st.subheader("Prediction Result")

# Check if the model is loaded before attempting to use it
if model is None:
    st.error("Model file 'loan_approval_model.joblib' not found. Please ensure it is in the same directory as the app.")
else:
    if st.sidebar.button("Predict Loan Status"):
        prediction = model.predict(user_input_df)
        prediction_proba = model.predict_proba(user_input_df)

        if prediction[0] == 1:
            st.success("🎉 Congratulations! Your loan is likely to be **Approved**.")
            st.write(f"**Confidence:** {prediction_proba[0][1] * 100:.2f}%")
        else:
            st.error("😔 We're sorry. Your loan is likely to be **Rejected**.")
            st.write(f"**Confidence:** {prediction_proba[0][0] * 100:.2f}%")

        with st.expander("See input data for model"):
            st.write("The following processed data was used for prediction:")
            st.dataframe(user_input_df)