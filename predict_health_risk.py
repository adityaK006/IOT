import pandas as pd
import joblib

# Load the trained model, scaler, and label encoders
model = joblib.load('student_health_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define feature columns (same order as in training)
feature_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Activity_Level', 'Medical_History', 
                   'Stress_Level', 'Sleep_Duration', 'Exercise_Habit', 'Caffeine_Intake', 'Study_Hours']

def get_user_data_and_predict():
    print("Please enter the following details:")

    # Get user input
    age = float(input("Age: "))
    weight = float(input("Weight: "))
    stress_level = input("Stress Level (Low/Medium/High): ")
    sleep_duration = float(input("Sleep Duration (hours): "))
    exercise_habit = input("Exercise Habit (Regular/Occasional/Rarely): ")

    # Ask for additional inputs
    activity_level = input("Activity Level (Resting/Walking/Running): ")
    medical_history = input("Medical History (No History/Diabetes/Cardiac Risk): ")

    # Convert categorical inputs to encoded values
    stress_level_encoded = label_encoders['Stress_Level'].transform([stress_level])[0]
    exercise_habit_encoded = label_encoders['Exercise_Habit'].transform([exercise_habit])[0]
    activity_level_encoded = label_encoders['Activity_Level'].transform([activity_level])[0]
    medical_history_encoded = label_encoders['Medical_History'].transform([medical_history])[0]

    # Construct DataFrame for prediction
    user_data = pd.DataFrame({
        'Age': [age],
        'Weight': [weight],
        'Height': [170],  # Default value (can be customized if needed)
        'Heart_Rate': [75],  # Default value (can be customized if needed)
        'Activity_Level': [activity_level_encoded],
        'Medical_History': [medical_history_encoded],
        'Stress_Level': [stress_level_encoded],
        'Sleep_Duration': [sleep_duration],
        'Exercise_Habit': [exercise_habit_encoded],
        'Caffeine_Intake': [label_encoders['Caffeine_Intake'].transform(['None'])[0]],  # Default value
        'Study_Hours': [4]  # Default value (can be customized if needed)
    })

    # Ensure the columns in user_data are in the same order as the trained model
    user_data = user_data[X.columns]

    # Normalize the numerical features for prediction
    user_data[['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']] = scaler.transform(
        user_data[['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']]
    )

    # Make a prediction
    user_prediction = model.predict(user_data)

    # Decode the predicted risk level
    predicted_risk = label_encoders['Risk_Level'].inverse_transform(user_prediction)[0]
    print(f"\nPredicted Risk Level: {predicted_risk}")
