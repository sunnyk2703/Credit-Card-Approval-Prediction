# Credit-Card-Approval-Prediction
Machine Learning project for predicting credit card approvals using classification algorithms, data preprocessing, feature engineering, and model evaluation for automated credit risk assessment.
💳 SmartLender: AI-Powered Credit Card Approval System
SmartLender is an AI-powered Credit Card Approval Prediction System that automates credit card eligibility decisions using machine learning. The model is trained on historical applicant information such as income, employment status, education, marital status, credit history, and other financial details to predict whether a credit card application should be Approved or Rejected.

The system minimizes manual verification, reduces processing time, and helps financial institutions make fast, consistent, and data-driven decisions through an intuitive Flask web application.

Machine Learning Models
The following classification algorithms are trained and evaluated:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
XGBoost Classifier
The best-performing model is selected and deployed using Flask. An optional IBM Watson Machine Learning deployment is also supported for cloud-based inference.

📂 Repository Structure
SmartLender/
├── Dataset/
│   ├── application_record.csv
│   └── credit_record.csv
│
├── Flask/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── model/
│       ├── smartlender_tuned_xgboost.pkl
│       ├── model_columns.pkl
│       ├── scaler.pkl
│       └── encoders.pkl
│
├── Project_Documentation/
│   ├── 1.Brainstorming_&_Ideation/
│   ├── 2.Requirement_Analysis/
│   ├── 3.Project_Design_Phase/
│   ├── 4.Project_Planning_Phase/
│   ├── 5.Project_Development_Phase/
│   ├── 6.Project_Testing/
│   ├── 7.Project_Documentation/
│   └── 8.Project_Demonstration/
│
├── Training/
│   └── SmartLender_Model_Training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
🚀 Getting Started
1. Clone the Repository
git clone https://github.com/your-username/SmartLender.git
cd SmartLender
2. Install Dependencies
pip install -r requirements.txt
3. Add the Dataset
Place the following datasets inside the Dataset/ folder:

application_record.csv
credit_record.csv
These datasets contain customer demographic information and historical credit records used for model training.

4. Train the Model
Open and execute:

Training/SmartLender_Model_Training.ipynb
The notebook performs:

Data Cleaning
Feature Engineering
Exploratory Data Analysis
Model Training
Hyperparameter Tuning
Model Evaluation
After training, the following files are generated:

smartlender_tuned_xgboost.pkl
model_columns.pkl
scaler.pkl
encoders.pkl
Copy these files into:

Flask/model/
5. Run the Flask Application
cd Flask
python app.py
Open your browser and visit:

http://127.0.0.1:5000
✨ Features
Credit Card Approval Prediction
Data Preprocessing & Feature Engineering
Multiple ML Model Comparison
XGBoost-based Prediction Engine
Responsive Flask Web Interface
Fast Real-Time Predictions
Model Serialization using Pickle
Optional IBM Watson ML Deployment
📊 Project Workflow
Data Collection
Data Cleaning
Exploratory Data Analysis
Feature Engineering
Model Training
Model Evaluation
Model Selection
Flask Deployment
Prediction
💳 Use Cases
Credit Card Eligibility Assessment
Predict whether a customer qualifies for a credit card based on financial and demographic information.

Risk Assessment
Identify applicants with higher default risk before approval.

Automated Customer Screening
Reduce manual verification effort by instantly classifying applications.

Financial Decision Support
Assist banks and financial institutions in making faster, more accurate approval decisions.

🤖 Machine Learning Pipeline
Data Cleaning
Missing Value Handling
Feature Encoding
Feature Scaling
Model Training
Hyperparameter Optimization
Performance Evaluation
Model Deployment
☁️ Optional IBM Watson Machine Learning Deployment
The trained XGBoost model can be deployed to IBM Watson Machine Learning using the ibm-watson-machine-learning SDK.

Once deployed, the Flask application can consume predictions through REST APIs instead of loading the local model, enabling scalable cloud-based inference.

🛠️ Technologies Used
Programming Language
Python
Machine Learning
Scikit-learn
XGBoost
NumPy
Pandas
Data Visualization
Matplotlib
Seaborn
Web Framework
Flask
HTML
CSS
Deployment
IBM Watson Machine Learning (Optional)
📈 Models Compared
Model	Purpose
Logistic Regression	Baseline Classification
Decision Tree	Rule-Based Prediction
Random Forest	Ensemble Learning
XGBoost	High-Performance Gradient Boosting
🎯 Future Enhancements
Deep Learning Models
Explainable AI (SHAP/LIME)
REST API Integration
User Authentication
Database Support
Cloud Deployment (AWS/Azure/IBM Cloud)
👨‍💻 Skills Demonstrated
Machine Learning
Data Analysis
Data Preprocessing
Feature Engineering
Classification Algorithms
XGBoost
Flask Development
Model Deployment
Python Programming
Scikit-learn
NumPy
Pandas
Matplotlib
Git & GitHub
📄 License
This project is developed for educational and academic purposes.
