import streamlit as st
import pandas as pd
import pickle
import json
import os

st.set_page_config(page_title='🧑‍💼 Employee Attrition Prediction', layout='wide')
st.title('🧑‍💼 Employee Attrition Prediction App')

MODEL_FILE = 'model.pkl'
FEATURE_FILE = 'feature_column.json'

# --- Load Model and Feature Columns ---
model = None
model_columns = []
model_files_loaded = False

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
else:
    st.error(f'❌ Model file not found: {MODEL_FILE}')

if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, 'r') as f:
        feature_data = json.load(f)
        model_columns = feature_data.get('columns')
    if model_columns:
        model_files_loaded = True
    else:
        st.error(f'❌ "columns" key not found in {FEATURE_FILE}')
else:
    st.error(f'❌ Feature list file not found: {FEATURE_FILE}')

if model_files_loaded and model:
    st.success('✅ Model and feature columns loaded successfully!')

# --- Define Input Options ---
# These must match the raw 'attrition.csv' columns
options = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
    'Gender': ['Male', 'Female'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 
                'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['Yes', 'No']
}

# --- Preprocessing Configuration (for server) ---
categorical_cols_server = [
    'BusinessTravel', 
    'Department', 
    'EducationField', 
    'JobRole', 
    'MaritalStatus'
]

binary_mappings_server = {
    'Gender': {'Male': 1, 'Female': 0},
    'OverTime': {'Yes': 1, 'No': 0}
}


# --- Render Input Form ---
st.header('Enter Employee Details:')

# Create columns for layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    Age = st.number_input('Age', min_value=18, max_value=100, value=35)
    Gender = st.selectbox('Gender', options['Gender'])
    MaritalStatus = st.selectbox('MaritalStatus', options['MaritalStatus'])
    DistanceFromHome = st.number_input('DistanceFromHome (miles)', min_value=1, max_value=30, value=10)

with col2:
    st.subheader("Job Details")
    JobRole = st.selectbox('JobRole', options['JobRole'])
    JobLevel = st.slider('JobLevel', 1, 5, 2)
    JobInvolvement = st.slider('JobInvolvement', 1, 4, 3)
    JobSatisfaction = st.slider('JobSatisfaction', 1, 4, 3)
    Department = st.selectbox('Department', options['Department'])
    
with col3:
    st.subheader("Compensation & Work")
    MonthlyIncome = st.number_input('MonthlyIncome ($)', min_value=1000, max_value=20000, value=5000)
    PercentSalaryHike = st.slider('PercentSalaryHike (%)', 11, 25, 15)
    OverTime = st.radio('OverTime', options['OverTime'])
    BusinessTravel = st.selectbox('BusinessTravel', options['BusinessTravel'])

with st.expander("Show All Fields (Advanced)"):
    # Layout for remaining fields
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Rates")
        DailyRate = st.number_input('DailyRate', min_value=100, max_value=1500, value=800)
        HourlyRate = st.number_input('HourlyRate', min_value=30, max_value=100, value=65)
        MonthlyRate = st.number_input('MonthlyRate', min_value=2000, max_value=27000, value=14000)

    with c2:
        st.subheader("Background")
        Education = st.slider('Education (1-5)', 1, 5, 3)
        EducationField = st.selectbox('EducationField', options['EducationField'])
        NumCompaniesWorked = st.number_input('NumCompaniesWorked', min_value=0, max_value=10, value=1)
        TotalWorkingYears = st.number_input('TotalWorkingYears', min_value=0, max_value=40, value=10)

    with c3:
        st.subheader("Company & Team")
        EnvironmentSatisfaction = st.slider('EnvironmentSatisfaction (1-4)', 1, 4, 3)
        PerformanceRating = st.slider('PerformanceRating (1-4)', 1, 4, 3)
        RelationshipSatisfaction = st.slider('RelationshipSatisfaction (1-4)', 1, 4, 3)
        StockOptionLevel = st.slider('StockOptionLevel', 0, 3, 1)
        TrainingTimesLastYear = st.number_input('TrainingTimesLastYear', min_value=0, max_value=6, value=3)
        WorkLifeBalance = st.slider('WorkLifeBalance (1-4)', 1, 4, 3)
        YearsAtCompany = st.number_input('YearsAtCompany', min_value=0, max_value=40, value=5)
        YearsInCurrentRole = st.number_input('YearsInCurrentRole', min_value=0, max_value=20, value=2)
        YearsSinceLastPromotion = st.number_input('YearsSinceLastPromotion', min_value=0, max_value=15, value=1)
        YearsWithCurrManager = st.number_input('YearsWithCurrManager', min_value=0, max_value=20, value=3)

# --- Collect Data on Button Click ---
if st.button('🧑‍💼 Predict Attrition'):
    if not model_files_loaded or not model:
        st.error('Model or feature list not loaded. Cannot predict.')
    else:
        # 1. Create a dictionary of the input data
        raw_data = {
            'Age': Age, 'BusinessTravel': BusinessTravel, 'DailyRate': DailyRate, 
            'Department': Department, 'DistanceFromHome': DistanceFromHome, 
            'Education': Education, 'EducationField': EducationField, 
            'EnvironmentSatisfaction': EnvironmentSatisfaction, 'Gender': Gender, 
            'HourlyRate': HourlyRate, 'JobInvolvement': JobInvolvement, 
            'JobLevel': JobLevel, 'JobRole': JobRole, 
            'JobSatisfaction': JobSatisfaction, 'MaritalStatus': MaritalStatus, 
            'MonthlyIncome': MonthlyIncome, 'MonthlyRate': MonthlyRate, 
            'NumCompaniesWorked': NumCompaniesWorked, 'OverTime': OverTime, 
            'PercentSalaryHike': PercentSalaryHike, 'PerformanceRating': PerformanceRating, 
            'RelationshipSatisfaction': RelationshipSatisfaction, 
            'StockOptionLevel': StockOptionLevel, 'TotalWorkingYears': TotalWorkingYears, 
            'TrainingTimesLastYear': TrainingTimesLastYear, 
            'WorkLifeBalance': WorkLifeBalance, 'YearsAtCompany': YearsAtCompany, 
            'YearsInCurrentRole': YearsInCurrentRole, 
            'YearsSinceLastPromotion': YearsSinceLastPromotion, 
            'YearsWithCurrManager': YearsWithCurrManager
        }
        
        # 2. Convert to DataFrame
        df = pd.DataFrame([raw_data])
        
        st.subheader("Input Data Preview")
        st.dataframe(df)

        # --- 3. Replicate Preprocessing from Flask App ---
        
        # 3a. Map binary string columns
        for col, mapping in binary_mappings_server.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # 3b. Apply one-hot encoding
        try:
            df_processed = pd.get_dummies(df, columns=categorical_cols_server, drop_first=True)
        except Exception as e:
            st.error(f"Error during one-hot encoding: {e}")
            st.stop()

        # 3c. Align columns with model_columns
        # This ensures all columns the model expects are present and in the correct order
        
        # Create a DataFrame with all model columns, initialized to 0
        df_aligned = pd.DataFrame(columns=model_columns)
        
        # Concatenate to align. Fills missing columns with 0 and keeps only model columns.
        df_aligned = pd.concat([df_aligned, df_processed], join='outer', sort=False)[model_columns]
        
        # Fill any NaNs that resulted from alignment
        df_aligned.fillna(0, inplace=True)
        
        # Ensure all dtypes are numeric
        df_aligned = df_aligned.astype(float)
        
        # st.subheader("Processed Data for Model")
        # st.dataframe(df_aligned)

        # --- 4. Make Prediction ---
        try:
            prediction = model.predict(df_aligned)[0]
            prediction_proba = model.predict_proba(df_aligned)[0]

            # --- 5. Format Output ---
            st.subheader('Prediction Result')
            if prediction == 1:
                st.error('Prediction: **Attrition (Yes)**', icon="⚠️")
                st.markdown(f"Probability of Attrition: **{prediction_proba[1]*100:.2f}%**")
            else:
                st.success('Prediction: **No Attrition (No)**', icon="✅")
                st.markdown(f"Probability of Attrition: **{prediction_proba[1]*100:.2f}%**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
