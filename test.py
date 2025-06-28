# #Working... 4th march 2025

# import joblib
# import pandas as pd

# # Load the model, scaler, and label encoders
# try:
#     model = joblib.load('student_health_risk_model2.pkl')
#     scaler = joblib.load('scaler2.pkl')
#     label_encoders = joblib.load('label_encoders2.pkl')

#     print("Model, scaler, and label encoders loaded successfully!")
# except Exception as e:
#     print(f"Error loading files: {e}")
#     exit()

# # Take user input dynamically
# try:
#     # Get inputs from the user
#     age = int(input("Enter Age: "))
#     weight = float(input("Enter Weight (kg): "))
#     stress_level = input("Enter Stress Level (Low/Medium/High): ")
#     sleep_duration = float(input("Enter Sleep Duration (hours): "))
#     exercise_habit = input("Enter Exercise Habit (Regular/Occasional/Rarely): ")
#     activity_level = input("Enter Activity Level (Resting/Walking/Running): ")
#     medical_history = input("Enter Medical History (No History/Diabetes/Cardiac Risk): ")

#     # Default values for optional inputs
#     height = 170  # Default height in cm
#     heart_rate = 75  # Default heart rate in bpm
#     caffeine_intake = 'None'  # Default caffeine intake
#     study_hours = 4  # Default study hours

#     # Create a DataFrame for the input data
#     sample_input = pd.DataFrame({
#         'Age': [age],
#         'Weight': [weight],
#         'Height': [height],
#         'Heart_Rate': [heart_rate],
#         'Activity_Level': [label_encoders['Activity_Level'].transform([activity_level])[0]],
#         'Medical_History': [label_encoders['Medical_History'].transform([medical_history])[0]],
#         'Stress_Level': [label_encoders['Stress_Level'].transform([stress_level])[0]],
#         'Sleep_Duration': [sleep_duration],
#         'Exercise_Habit': [label_encoders['Exercise_Habit'].transform([exercise_habit])[0]],
#         'Caffeine_Intake': [label_encoders['Caffeine_Intake'].transform([caffeine_intake])[0]],
#         'Study_Hours': [study_hours]
#     })

#     # Ensure the columns in sample_input match the order in the training data
#     sample_input = sample_input[model.feature_names_in_]

#     # Normalize the numerical features only
#     numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
#     sample_input[numerical_columns] = scaler.transform(sample_input[numerical_columns])

#     # Make a prediction
#     prediction = model.predict(sample_input)
#     predicted_risk = label_encoders['Risk_Level'].inverse_transform(prediction)[0]

#     print(f"\nPredicted Risk Level: {predicted_risk}")

# except Exception as e:
#     print(f"Error during prediction: {e}")




import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the trained model, scaler, and label encoders
model = joblib.load('student_health_risk_model2.pkl')
scaler = joblib.load('scaler2.pkl')
label_encoders = joblib.load('label_encoders2.pkl')

# Load the unseen test data
df = pd.read_csv('E:/COLLEGE/IOT/IOT project/unseen_test_data.csv')

# Encode categorical features
categorical_cols = ['Activity_Level', 'Medical_History', 'Stress_Level', 
                    'Exercise_Habit', 'Caffeine_Intake']

for col in categorical_cols:
    df[col] = label_encoders[col].transform(df[col])

# Normalize numerical features
numerical_cols = ['Age', 'Weight', 'Height', 'Heart_Rate', 
                  'Sleep_Duration', 'Study_Hours']

df[numerical_cols] = scaler.transform(df[numerical_cols])

# ✅ Fix the Feature Order (Very Important)
# This line will force the column order to match the model's training data
df = df[model.feature_names_in_]

# ✅ Make predictions without error
predictions = model.predict(df)

# Decode the predictions back to Risk_Level
df['Predicted_Risk_Level'] = label_encoders['Risk_Level'].inverse_transform(predictions)

# ✅ Save the results to a CSV file
output_path = 'E:/COLLEGE/IOT/IOT project/prediction_results.csv'
df.to_csv(output_path, index=False)

print("\n✅ Predictions saved successfully to:", output_path)
print("\n✅ Showing Predictions:\n")
print(df[['Age', 'Weight', 'Height', 'Heart_Rate', 
          'Activity_Level', 'Medical_History', 
          'Stress_Level', 'Sleep_Duration', 
          'Exercise_Habit', 'Predicted_Risk_Level']])

# ----- OPTIONAL: Test Accuracy if true labels exist -----
# If you manually labeled the test data, uncomment this
# y_true = label_encoders['Risk_Level'].transform(df['Risk_Level']) # Assuming true labels exist
# accuracy = accuracy_score(y_true, predictions)
# print(f"\n✅ Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# Generate classification report
print("\n✅ Classification Report:\n")
print(classification_report(predictions, predictions))

# Generate confusion matrix
print("\n✅ Confusion Matrix:\n")
print(confusion_matrix(predictions, predictions))


