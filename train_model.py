print("scripted started")
import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("application_record.csv")   # Change to your CSV filename if needed

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Drop Loan_ID
df.drop("Loan_ID", axis=1, inplace=True)

# Target
y = df["Loan_Status"]
X = df.drop("Loan_Status", axis=1)

# Encode categorical columns
encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
encoders["Loan_Status"] = target_encoder

# Save column names
model_columns = list(X.columns)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_scaled, y)

# Create folder
os.makedirs("Flask/model", exist_ok=True)

# Save files
joblib.dump(model, "Flask/model/smartlender_tuned_xgboost.pkl")
joblib.dump(scaler, "Flask/model/scaler.pkl")
joblib.dump(encoders, "Flask/model/encoders.pkl")
joblib.dump(model_columns, "Flask/model/model_columns.pkl")

print("All model files created successfully!")