import math
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

with open("model/smartlender_tuned_xgboost.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/model_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

model_name = "XGBoost (tuned)"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None, model_name=model_name)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    applicant_income = float(form.get("applicant_income", 0))
    coapplicant_income = float(form.get("coapplicant_income", 0))
    loan_amount = float(form.get("loan_amount", 0))

    raw_input = {
        "Gender": form.get("gender"),
        "Married": form.get("married"),
        "Dependents": form.get("dependents"),
        "Education": form.get("education"),
        "Self_Employed": form.get("self_employed"),
        "Loan_Amount_Term": float(form.get("loan_amount_term", 360)),
        "Credit_History": float(form.get("credit_history", 1)),
        "Property_Area": form.get("property_area"),
        "LoanAmount_log": math.log1p(loan_amount),
        "Total_Income_log": math.log1p(applicant_income + coapplicant_income),
    }

    df_input = pd.DataFrame([raw_input])

    for col, le in encoders.items():
        if col in df_input.columns and col != "Loan_Status":
            val = df_input.at[0, col]
            if val in le.classes_:
                df_input[col] = le.transform([val])
            else:
                df_input[col] = 0

    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    X_scaled = scaler.transform(df_input)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    result = "Approved" if pred == 1 else "Rejected"
    confidence = round((prob if pred == 1 else 1 - prob) * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        model_name=model_name,
    )


if __name__ == "__main__":
    app.run(debug=True)
