README: Heart rate Monitoring system: Analysis of Herat Rate
Overview
This model is designed to predict the health risk level of users based on various factors such as age, weight, height, sleep duration, stress level, and medical history. The model is trained using a Random Forest Classifier and incorporates feature encoding and normalization for better prediction accuracy.

Requirements
Before using the model, make sure you have the following installed:

Python 3.x
Libraries:
pandas
numpy
scikit-learn
joblib

You can install the necessary libraries using the following commands:
pip install pandas numpy scikit-learn joblib

Files Included:
student_health_risk_model.pkl: The trained Random Forest model for predicting student health risk.
scaler.pkl: The saved StandardScaler used for normalizing input features.
label_encoders.pkl: The saved LabelEncoders used to encode categorical features.

Model Details:
Dataset: The model is trained on a dataset that includes features such as Age, Weight, Height, Heart Rate, Activity Level, Stress Level, Sleep Duration, and Exercise Habit.
Target Variable: The target variable is the Risk Level, which is encoded into numeric values.

The following categorical features are encoded using LabelEncoder:

Activity_Level
Medical_History
Stress_Level
Exercise_Habit
Caffeine_Intake
Risk_Level

Steps to Use the Model
1. Load the Model
To load the pre-trained model and related artifacts, use the following code:

python
Copy code
import joblib

# Load the trained model, scaler, and label encoders
model = joblib.load('student_health_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

2. Get User Input and Make Predictions-
Now call the existing function that takes input and makes the prediction get_user_data_and_predict(). This will automatically call the input and prediction process. The function will return the Predicted Risk Level based on the user's input.
The model includes a predefined function get_user_data_and_predict() to take input from the user and make a prediction. It will prompt the user for the following details:
Age
Weight
Stress Level (Low/Medium/High)
Sleep Duration (hours)
Exercise Habit (Regular/Occasional/Rarely)
Activity Level (Resting/Walking/Running)
Medical History (No History/Diabetes/Cardiac Risk)


3. Example of Input and Prediction-
Here is an example of how the input will be taken from the user:
Please enter the following details:
Age: 20
Weight: 65
Stress Level (Low/Medium/High): High
Sleep Duration (hours): 4
Exercise Habit (Regular/Occasional/Rarely): Rarely
Activity Level (Resting/Walking/Running): Resting
Medical History (No History/Diabetes/Cardiac Risk): Cardiac Risk
The model will then return the following output:

Predicted Risk Level: High

4. Model Testing and Usage-
If you want to test the model by loading it and making a prediction, you can follow these steps:
Load the Model, Scaler, and Label Encoders using joblib.load.
Call the get_user_data_and_predict() function to take user input and make a prediction.

import joblib
# Load the trained model, scaler, and label encoders
model = joblib.load('student_health_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Now call the existing function that takes input and makes the prediction
get_user_data_and_predict()  # This will automatically call the input and prediction process

5. Model Components-
Model: A Random Forest Classifier trained to predict the health risk level.
Scaler: A StandardScaler used to normalize numerical features (Age, Weight, Height, etc.).
Label Encoders: LabelEncoders used to encode categorical features like Stress Level, Activity Level, etc.


Conclusion-
Heart rate Monitoring system: Analysis of Herat Rate is a simple yet powerful tool to predict the health risk level based on a users lifestyle and medical history. You can easily use the model by loading it, entering the required data, and obtaining predictions for the student's health risk level.