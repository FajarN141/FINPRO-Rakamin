import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Fungsi custom feature engineering
def custom_feature_engineering(df):
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

    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 70], labels=['18-30', '30-40', '40-50', '50-60', '60+'])
    df['DistanceGroup'] = pd.cut(df['DistanceFromHome'], bins=[0, 5, 10, 20, 30], labels=['0-5', '5-10', '10-20', '20+'])

    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    return df

# 2. Load raw data
data_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"  # ubah jika path beda
df_raw = pd.read_csv(data_path)
y = df_raw['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# 3. Pipeline
categorical_cols = [
    'Department', 'EducationField', 'Gender', 'JobRole',
    'MaritalStatus', 'OverTime', 'AgeGroup', 'DistanceGroup',
    'BusinessTravel'
]
text_ordinal_cols = [
    'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction',
    'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance'
]

# Gunakan fitur setelah custom_feature_engineering
df_temp = custom_feature_engineering(df_raw)
X_temp = df_temp.drop('Attrition', axis=1)
numerical_cols = [col for col in X_temp.columns if col not in categorical_cols + text_ordinal_cols]

# 4. Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols + text_ordinal_cols)
])

# 5. Full pipeline
pipeline = Pipeline([
    ('feature_eng', FunctionTransformer(custom_feature_engineering, validate=False)),
    ('preprocess', preprocessor),
    ('model', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
    ))
])

# 6. Fit pipeline
pipeline.fit(df_raw, y)

# 7. Simpan
joblib.dump(pipeline, "adaboost_pipeline_kosongan.pkl")
print("✅ Pipeline berhasil disimpan sebagai adaboost_pipeline.pkl")
