import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the Dataset
df = pd.read_csv('hospital_charges.csv')

# Step 2: Exploratory Data Analysis (EDA)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())  # Check for missing values

# Visualize distribution of hospital charges
plt.figure(figsize=(8, 6))
sns.histplot(df['charges'], bins=30, kde=True)
plt.title("Distribution of Hospital Charges")
plt.xlabel("Charges ($)")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Boxplot of charges by smoking status
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Hospital Charges by Smoking Status")
plt.show()

# Step 3: Data Preprocessing
# Handle missing values
df = df.dropna()

# Feature Engineering: Create new interaction features
df['bmi_smoker'] = df['bmi'] * df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
df['age_children'] = df['age'] * df['children']

# Define categorical and numerical features
categorical_features = ['sex', 'region', 'smoker']
numerical_features = ['age', 'bmi', 'children', 'bmi_smoker', 'age_children']

# Column transformer for preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Step 4: Define Features and Target
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define Machine Learning Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Step 6: Train, Evaluate, and Compare Models
results = {}
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

# Convert results to DataFrame
result_df = pd.DataFrame(results).T

# Plot Model Performance
plt.figure(figsize=(8, 6))
sns.barplot(x=result_df.index, y='R2', data=result_df)
plt.title("Model Performance Comparison (R2 Score)")
plt.ylabel("R2 Score")
plt.show()

# Step 7: Predict New Data
new_data = pd.DataFrame({
    'age': [40],
    'sex': ['male'],
    'bmi': [30.5],
    'children': [2],
    'smoker': ['yes'],
    'region': ['southeast'],
    'bmi_smoker': [30.5],
    'age_children': [40 * 2]
})

best_model = models['XGBoost']
pipeline = Pipeline([('preprocessor', preprocessor), ('model', best_model)])
pipeline.fit(X_train, y_train)
predicted_charge = pipeline.predict(new_data)
print(f"Predicted Hospital Charge: ${predicted_charge[0]:.2f}")
