import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('E:\COLLEGE\IOT\IOT project\balanced_student_health_dataset.csv')

# Encode categorical features
categorical_cols = ['Activity_Level', 'Medical_History', 'Stress_Level', 'Exercise_Habit', 'Caffeine_Intake', 'Risk_Level']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Ensure 'None' is included as a valid label for caffeine intake
label_encoders['Caffeine_Intake'].classes_ = np.append(label_encoders['Caffeine_Intake'].classes_, 'None')

# Define features (X) and target (y)
X = df.drop(columns=['Risk_Level'])
y = df['Risk_Level']

# Normalize numerical features
scaler = StandardScaler()
X[['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']] = scaler.fit_transform(
    X[['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']]
)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model, scaler, and label encoders
joblib.dump(model, 'student_health_risk_model2.pkl')
joblib.dump(scaler, 'scaler2.pkl')
joblib.dump(label_encoders, 'label_encoders2.pkl')

print("\nModel, scaler, and label encoders saved successfully.")

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

