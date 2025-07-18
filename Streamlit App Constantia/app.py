import streamlit as st
import pandas as pd
import joblib

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
    return "Resign" if prob >= threshold else "Tidak Resign"

# 🚀 Load pipeline
pipeline = joblib.load("adaboost_pipeline_kosongan.pkl")

# 🎯 Streamlit Layout
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("🔍 Employee Attrition Prediction")

# 🎛 Mode input
mode = st.radio("Pilih Mode Prediksi", ["Individu", "Batch (CSV)"])

if mode == "Individu":
    st.subheader("📥 Masukkan Data Karyawan")

    with st.form("form_karyawan"):
        col1, col2 = st.columns(2)

        with col1:
            Age = st.number_input("Age", 18, 60, 30)
            DistanceFromHome = st.number_input("Distance From Home (KM)", 0, 30, 5)
            MonthlyIncome = st.number_input("Gaji Bulanan", 1000, 20000, 5000)
            DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
            HourlyRate = st.number_input("Hourly Rate", 30, 100, 60)
            MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 20000)
            OverTime = st.selectbox("Overtime", ['Yes', 'No'])
            JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
            EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
            WorkLifeBalance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
            Department = st.selectbox("Departemen", ['Research & Development', 'Sales', 'Human Resources'])
            EducationField = st.selectbox("Education Field", ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
            BusinessTravel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])


        with col2:
            NumCompaniesWorked = st.number_input("Jumlah Perusahaan Sebelumnya", 0, 10, 2)
            TotalWorkingYears = st.number_input("Total Tahun Bekerja", 0, 40, 5)
            YearsAtCompany = st.number_input("Years at Company", 0, 40, 3)
            YearsInCurrentRole = st.number_input("Years In Current Role", 0, 40, 2)
            YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
            YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 15, 2)
            TrainingTimesLastYear = st.number_input("Training Time Last Year", 0, 10, 3)
            PercentSalaryHike = st.number_input("Persentase Kenaikan Gaji", 0, 100, 12)
            JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            Education = st.selectbox("Education", [1, 2, 3, 4, 5])
            JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
            PerformanceRating = st.selectbox("Performance Rating", [1, 2, 3, 4])
            RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
            Gender = st.selectbox("Gender", ['Male', 'Female'])
            JobRole = st.selectbox("Job Role", [
                'Sales Executive', 'Research Scientist', 'Laboratory Technician',
                'Manufacturing Director', 'Healthcare Representative',
                'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
            ])
            MaritalStatus = st.selectbox("Status Pernikahan", ['Single', 'Married', 'Divorced'])
            StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])

        submitted = st.form_submit_button("Prediksi Sekarang")

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
            'EmployeeNumber': 1234
        }])

        prob = pipeline.predict_proba(data)[0][1]
        status = interpret_result(prob)
        st.success(f"📊 Kemungkinan Resign: {prob:.2%} → **{status}**")

# ========== BATCH MODE ==========
else:
    st.subheader("📤 Unggah File CSV")
    uploaded_file = st.file_uploader("Upload file csv sesuai format", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Preview Data:", df.head())

        if st.button("Prediksi Batch"):
            probs = pipeline.predict_proba(df)[:, 1]
            hasil = pd.DataFrame({
                "Kemungkinan Resign (%)": probs * 100,
                "Status": ["Resign" if p >= 0.49 else "Tidak Resign" for p in probs]
            })
            output = pd.concat([df.reset_index(drop=True), hasil], axis=1)
            st.dataframe(output)
            st.download_button("💾 Download Hasil Prediksi", output.to_csv(index=False), "hasil_prediksi.csv", "text/csv")
