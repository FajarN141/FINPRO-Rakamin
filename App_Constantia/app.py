import streamlit as st
import pandas as pd
import io
import joblib

SATISFACTION_LEVELS = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
EDUCATION_LEVELS = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
WORK_LIFE_BALANCE = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
PERFORMANCE_RATINGS = {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"}

DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"]
JOB_ROLES = [
    "Healthcare Representative", "Human Resources", "Laboratory Technician",
    "Manager", "Manufacturing Director", "Research Director", 
    "Research Scientist", "Sales Executive", "Sales Representative"
]
EDUCATION_FIELDS = [
    "Human Resources", "Life Sciences", "Marketing", 
    "Medical", "Technical Degree", "Other"
]

def personal_info_section():
    """Personal Information Input Section"""
    st.subheader("👤 Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        employee_number = st.text_input("Employee Number", value="1234", help="Unique identifier for the employee")
        age = st.number_input("Age", min_value=18, max_value=70, value=35)
        gender = st.selectbox("Gender", ["Female", "Male"])
        
    with col2:
        marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
        distance_from_home = st.number_input("Distance from Home (km)", min_value=0, max_value=50, value=10)
    
    return employee_number, age, gender, marital_status, distance_from_home

def job_info_section():
    """Job Information Input Section"""
    st.subheader("💼 Job Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        department = st.selectbox("Department", DEPARTMENTS)
        job_role = st.selectbox("Job Role", JOB_ROLES)
        job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2, 
                                help="1: Entry Level, 2: Mid Level, 3: Senior Level, 4: Manager, 5: Executive")
        
    with col2:
        business_travel = st.selectbox("Business Travel", ["None", "Rarely", "Frequently"])
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        stock_option_level = st.number_input("Stock Option Level", min_value=0, max_value=3, value=1, 
                                        help="0: No stock options, 1-3: Increasing levels of stock options")
    
    return department, job_role, job_level, business_travel, overtime, stock_option_level

def education_section():
    """Education & Training Input Section"""
    st.subheader("🎓 Education & Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5], 
                                format_func=lambda x: EDUCATION_LEVELS[x])
        education_field = st.selectbox("Education Field", EDUCATION_FIELDS)
        
    with col2:
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
    
    return education, education_field, training_times_last_year

def compensation_section():
    """Compensation Input Section"""
    st.subheader("💰 Compensation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=2000, value=800)
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
        
    with col2:
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=30000, value=15000)
        percent_salary_hike = st.number_input("Percent Salary Hike", min_value=10, max_value=30, value=15)
    
    return monthly_income, daily_rate, hourly_rate, monthly_rate, percent_salary_hike

def experience_section():
    """Work Experience Input Section"""
    st.subheader("📈 Work Experience")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=10)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=5)
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=50, value=3)
        
    with col2:
        years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=50, value=2)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=2)
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
    
    return total_working_years, years_at_company, years_in_current_role, years_with_curr_manager, years_since_last_promotion, num_companies_worked

def satisfaction_section():
    """Satisfaction & Performance Input Section"""
    st.subheader("😊 Satisfaction & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], 
                                        format_func=lambda x: SATISFACTION_LEVELS[x])
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], 
                                        format_func=lambda x: SATISFACTION_LEVELS[x])
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], 
                                        format_func=lambda x: SATISFACTION_LEVELS[x])
        
    with col2:
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], 
                                        format_func=lambda x: WORK_LIFE_BALANCE[x])
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], 
                                        format_func=lambda x: SATISFACTION_LEVELS[x])
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4], 
                                        format_func=lambda x: PERFORMANCE_RATINGS[x])
    
    return job_satisfaction, environment_satisfaction, relationship_satisfaction, work_life_balance, job_involvement, performance_rating

def display_prediction_results(prediction, probability):
    """Display prediction results in a formatted way"""
    st.markdown("---")
    st.subheader("🔮 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == "Resign":
            st.error("⚠️ **HIGH ATTRITION RISK**")
        else:
            st.success("✅ **LOW ATTRITION RISK**")
    
    with col2:
        if prediction == "Resign":
            st.error(f"📊 Probability of Resignation: {probability:.2%} → **{prediction}**")
        else:
            st.success(f"📊 Probability of Resignation: {probability:.2%} → **{prediction}**")
        

def custom_feature_engineering(df):
    import pandas as pd
    df = df.copy()
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

    ordinal_mapping = {
        'EnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'JobInvolvement': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'JobSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'PerformanceRating': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
        'RelationshipSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
        'WorkLifeBalance': {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
    }

    for col, mapping in ordinal_mapping.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df['IncomePerYear'] = df['MonthlyIncome'] * 12
    df['DailyRateToMonthlyRateRatio'] = df['DailyRate'] / df['MonthlyRate']
    df['HourlyRateToMonthlyRateRatio'] = df['HourlyRate'] / df['MonthlyRate']
    df['AvgYearsPerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
    df['PromotionRate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 0.001)
    df['RoleStability'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 0.001)
    df['CareerGrowth'] = df['JobLevel'] / (df['TotalWorkingYears'] + 0.001)
    df['ManagerStability'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 0.001)
    df['TrainingPerYear'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 0.001)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 70],
                            labels=['18-30', '30-40', '40-50', '50-60', '60+'])
    df['DistanceGroup'] = pd.cut(df['DistanceFromHome'], bins=[0, 5, 10, 20, 30],
                            labels=['0-5', '5-10', '10-20', '20+'])

    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    return df

# ⛑ Fungsi interpretasi hasil
def interpret_result(prob, threshold=0.49):
    return "Resign" if prob >= threshold else "Stay"

# 🚀 Load pipeline
pipeline = joblib.load("adaboost_pipeline.pkl")

# 🎯 Streamlit Layout
st.set_page_config(page_title="Employee Attrition Prediction - Constantia", layout="wide")
st.title("🔍 Employee Attrition Prediction")

# 🎛 Mode input
mode = st.radio("Choose Prediction Mode", ["Individual", "Batch (CSV/Excel)"])

if mode == "Individual":
    st.subheader("📥 Input Employee Data")

    with st.form("form_employee"):
        # Personal Information
        EmployeeNumber, Age, Gender, MaritalStatus, DistanceFromHome = personal_info_section()
        st.markdown("---")
            
        # Job Information
        Department, JobRole, JobLevel, BusinessTravel, OverTime, StockOptionLevel = job_info_section()
        st.markdown("---")
            
        # Education & Training
        Education, EducationField, TrainingTimesLastYear = education_section()
        st.markdown("---")
            
        # Compensation
        MonthlyIncome, DailyRate, HourlyRate, MonthlyRate, PercentSalaryHike = compensation_section()
        st.markdown("---")
            
        # Work Experience
        TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, YearsSinceLastPromotion, NumCompaniesWorked = experience_section()
        st.markdown("---")
            
        # Satisfaction & Performance
        JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance, JobInvolvement, PerformanceRating = satisfaction_section()

        # Submit button
        submitted = st.form_submit_button("Predict Now")

    if submitted:
        data = pd.DataFrame([{
            'Age': Age,
            'DistanceFromHome': DistanceFromHome,
            'DailyRate': DailyRate,
            'HourlyRate': HourlyRate,
            'BusinessTravel': BusinessTravel,
            'StockOptionLevel': StockOptionLevel,
            'MonthlyRate': MonthlyRate,
            'MonthlyIncome': MonthlyIncome,
            'NumCompaniesWorked': NumCompaniesWorked,
            'PercentSalaryHike': PercentSalaryHike,
            'TotalWorkingYears': TotalWorkingYears,
            'TrainingTimesLastYear': TrainingTimesLastYear,
            'YearsAtCompany': YearsAtCompany,
            'YearsInCurrentRole': YearsInCurrentRole,
            'YearsSinceLastPromotion': YearsSinceLastPromotion,
            'YearsWithCurrManager': YearsWithCurrManager,
            'JobLevel': JobLevel,
            'Education': Education,
            'EnvironmentSatisfaction': EnvironmentSatisfaction,
            'JobInvolvement': JobInvolvement,
            'JobSatisfaction': JobSatisfaction,
            'PerformanceRating': PerformanceRating,
            'RelationshipSatisfaction': RelationshipSatisfaction,
            'WorkLifeBalance': WorkLifeBalance,
            'OverTime': OverTime,
            'Department': Department,
            'EducationField': EducationField,
            'Gender': Gender,
            'JobRole': JobRole,
            'MaritalStatus': MaritalStatus,
            'Over18': 'Y',
            'EmployeeCount': 1,
            'StandardHours': 80,
            'EmployeeNumber': EmployeeNumber
        }])

        try:
            prob = pipeline.predict_proba(data)[0][1]
            status = interpret_result(prob)

            # Display results
            display_prediction_results(status, prob)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.write("Please ensure all inputs are valid and try again.")

# ========== BATCH MODE ==========
else:
    st.subheader("📤 Upload File")
    # File upload section
    uploaded_file = st.file_uploader("Upload file", type=['xlsx', 'xls', 'csv'], label_visibility="collapsed",
                                    help="Upload a CSV or Excel file containing employee data for batch prediction.")
    
    if uploaded_file is not None:
        # Load data based on file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("📄 Preview Data:", df.head())

        if st.button("Predict Batch"):
            st.markdown("---")
            st.subheader("🔮 Prediction Results")

            probs = pipeline.predict_proba(df)[:, 1]
            hasil = pd.DataFrame({
                "Probability of Resignation (%)": probs * 100,
                "Status": ["Resign" if p >= 0.49 else "Stay" for p in probs]
            })
            output = pd.concat([df.reset_index(drop=True), hasil], axis=1)
            st.dataframe(output)
            if uploaded_file.name.endswith('.csv'):
                st.download_button("💾 Download Prediction Results", output.to_csv(index=False), "prediction_results.csv", "text/csv")
            else:
                output_excel = io.BytesIO()
                with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                    output.to_excel(writer, index=False, sheet_name='Predictions')
                st.download_button("💾 Download Prediction Results", output_excel.getvalue(), "prediction_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # Generate and display dummy template
    st.markdown("---")
    st.markdown("### <i class='fa-solid fa-file-excel'></i> Excel Template", unsafe_allow_html=True)

    num_rows = 6  
    dummy_data = {}

    dummy_data['EmployeeNumber'] = [f"EMP{i+1}" for i in range(num_rows)]
    dummy_data['Age'] = [30, 35, 40, 45, 50, 55]
    dummy_data['DistanceFromHome'] = [5, 10, 15, 20, 25, 30]
    dummy_data['DailyRate'] = [500, 600, 700, 800, 900, 1000]
    dummy_data['HourlyRate'] = [50, 60, 70, 80, 90, 100]
    dummy_data['BusinessTravel'] = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
    dummy_data['StockOptionLevel'] = [0, 1, 2, 3, 0, 1]
    dummy_data['MonthlyRate'] = [2000, 2500, 3000, 3500, 4000, 4500]
    dummy_data['MonthlyIncome'] = [3000, 3500, 4000, 4500, 5000, 5500]
    dummy_data['NumCompaniesWorked'] = [1, 2, 3, 4, 1, 2]
    dummy_data['PercentSalaryHike'] = [10, 12, 14, 16, 18, 20]
    dummy_data['TotalWorkingYears'] = [5, 6, 7, 8, 9, 10]
    dummy_data['TrainingTimesLastYear'] = [1, 2, 3, 4, 5, 6]
    dummy_data['YearsAtCompany'] = [2, 3, 4, 5, 6, 7]
    dummy_data['YearsInCurrentRole'] = [1, 2, 3, 4, 5, 6]
    dummy_data['YearsSinceLastPromotion'] = [0, 1, 2, 3, 4, 5]
    dummy_data['YearsWithCurrManager'] = [1, 2, 3, 4, 5, 6]
    dummy_data['JobLevel'] = [1, 2, 3, 4, 5, 1]
    dummy_data['Education'] = [1, 2, 3, 4, 5, 1]
    dummy_data['EnvironmentSatisfaction'] = [1, 2, 3, 4, 1, 2]
    dummy_data['JobInvolvement'] = [1, 2, 3, 4, 1, 2]
    dummy_data['JobSatisfaction'] = [1, 2, 3, 4, 1, 2]
    dummy_data['PerformanceRating'] = [1, 2, 3, 4, 1, 2]
    dummy_data['RelationshipSatisfaction'] = [1, 2, 3, 4, 1, 2]
    dummy_data['WorkLifeBalance'] = [1, 2, 3, 4, 1, 2]
    dummy_data['OverTime'] = ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']
    dummy_data['Department'] = ['Research & Development', 'Sales', 'Human Resources', 'Research & Development', 'Sales', 'Human Resources']
    dummy_data['EducationField'] = ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources']
    dummy_data['Gender'] = ['Male', 'Female', 'Female', 'Male', 'Female', 'Male']   
    dummy_data['JobRole'] = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                            'Manufacturing Director', 'Healthcare Representative', 'Manager']
    dummy_data['MaritalStatus'] = ['Single', 'Married', 'Divorced', 'Single', 'Married', 'Divorced']
    dummy_data['EmployeeCount'] = [1, 1, 1, 1, 1, 1]
    dummy_data['Over18'] = ['Y', 'Y', 'Y', 'Y', 'Y', 'Y']
    dummy_data['StandardHours'] = [80, 80, 80, 80, 80, 80]

    df_dummy = pd.DataFrame(dummy_data)
    st.dataframe(df_dummy)

    # Create downloadable Excel template
    output_dummy = io.BytesIO()
    with pd.ExcelWriter(output_dummy, engine='xlsxwriter') as writer:
        df_dummy.to_excel(writer, index=False, sheet_name='Template')

    dummy_excel = output_dummy.getvalue()

    st.download_button(
        label="Download Example Template",
        data=dummy_excel,
        file_name='example_template.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, AdaBoost, and SMOTE - Constantia")
