import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# --- 1. Load Data ---
try:
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip().str.lower()
    if "loan_id" in df.columns:
        df = df.drop("loan_id", axis=1)
    df["loan_status"] = df["loan_status"].astype(str).str.strip().str.lower().map({"approved": 1, "rejected": 0})
except FileNotFoundError:
    st.error("Error: 'loan_approval_dataset.csv' not found in this folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Features & Target
X_full = df.drop("loan_status", axis=1)
y_full = df["loan_status"]

num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocessing
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# Train model once at startup
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
)
X_train_prep = preprocessor.fit_transform(X_train)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_prep, y_train)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_res, y_train_res)

def predict_loan_status(input_data):
    data_processed = preprocessor.transform(input_data)
    prediction = dt.predict(data_processed)
    probabilities = dt.predict_proba(data_processed)[0]
    return prediction[0], probabilities[1]

# --- 2. Streamlit App ---
st.set_page_config(page_title="Loan Eligibility Checker", layout="wide")
st.title("💰 Loan Eligibility Checker")
st.markdown("Check if your loan application is likely to be **approved** or **rejected**.")

st.markdown("---")

# Boundaries from dataset
max_income = df['income_annum'].max()
max_loan = df['loan_amount'].max()
max_assets = df[['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']].stack().max()

# Input Section
with st.container():
    st.header("📋 Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        income_annum = st.number_input(
            "Annual Income (₹)", 
            min_value=100000, 
            max_value=int(max_income) + 1000000, 
            value=500000, 
            step=50000
        )
        loan_amount = st.number_input(
            "Loan Amount Requested (₹)", 
            min_value=50000, 
            max_value=int(max_loan) + 1000000, 
            value=200000, 
            step=50000
        )
        loan_term = st.selectbox("Loan Term (Years)", options=sorted(df['loan_term'].unique()))
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=650, step=10)
        
    with col2:
        education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Employment Status", options=["Self Employed", "Job"])
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
        
        total_assets_value = st.number_input(
            "Total Assets Value (₹)", 
            min_value=0, 
            max_value=int(max_assets) * 2 + 5000000, 
            value=1000000, 
            step=50000
        )
        single_asset_value = total_assets_value / 4

# --- 3. Prediction ---
if st.button("Predict Loan Status", key="predict_btn", type="primary"):
    input_data = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [education],
        'self_employed': [self_employed],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [single_asset_value],
        'commercial_assets_value': [single_asset_value],
        'luxury_assets_value': [single_asset_value],
        'bank_asset_value': [single_asset_value]
    })
    
    input_data = input_data[X_full.columns]  # ensure correct order
    
    try:
        prediction, prob_approved = predict_loan_status(input_data)
        
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        if prediction == 1:
            st.balloons()
            st.success("✅ LOAN APPROVED!")
            st.metric("Approval Probability", f"{prob_approved:.2%}")
        else:
            st.snow()
            st.error("❌ LOAN REJECTED.")
            st.metric("Rejection Probability", f"{(1 - prob_approved):.2%}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
