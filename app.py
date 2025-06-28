# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd

# # Initialize the Flask application
# app = Flask(__name__)

# # Load the model, scaler, and label encoders
# try:
#     model = joblib.load('student_health_risk_model.pkl')
#     scaler = joblib.load('scaler.pkl')
#     label_encoders = joblib.load('label_encoders.pkl')
# except Exception as e:
#     raise Exception(f"Error loading files: {e}")

# # Define the home route
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Define the prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect data from the form
#         age = int(request.form['age'])
#         weight = float(request.form['weight'])
#         stress_level = request.form['stress_level']
#         sleep_duration = float(request.form['sleep_duration'])
#         exercise_habit = request.form['exercise_habit']
#         activity_level = request.form['activity_level']
#         medical_history = request.form['medical_history']

#         # Default values for optional inputs
#         height = 170  # Default height in cm
#         heart_rate = 75  # Default heart rate in bpm
#         caffeine_intake = 'None'  # Default caffeine intake
#         study_hours = 4  # Default study hours

#         # Prepare the input DataFrame
#         sample_input = pd.DataFrame({
#             'Age': [age],
#             'Weight': [weight],
#             'Height': [height],
#             'Heart_Rate': [heart_rate],
#             'Activity_Level': [label_encoders['Activity_Level'].transform([activity_level])[0]],
#             'Medical_History': [label_encoders['Medical_History'].transform([medical_history])[0]],
#             'Stress_Level': [label_encoders['Stress_Level'].transform([stress_level])[0]],
#             'Sleep_Duration': [sleep_duration],
#             'Exercise_Habit': [label_encoders['Exercise_Habit'].transform([exercise_habit])[0]],
#             'Caffeine_Intake': [label_encoders['Caffeine_Intake'].transform([caffeine_intake])[0]],
#             'Study_Hours': [study_hours]
#         })

#         # Ensure the columns match the training data
#         sample_input = sample_input[model.feature_names_in_]

#         # Normalize the numerical features
#         numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
#         sample_input[numerical_columns] = scaler.transform(sample_input[numerical_columns])

#         # Make the prediction
#         prediction = model.predict(sample_input)
#         predicted_risk = label_encoders['Risk_Level'].inverse_transform(prediction)[0]

#         # Return the result
#         return jsonify({'Predicted Risk Level': predicted_risk})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the model, scaler, and label encoders
try:
    model = joblib.load('student_health_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except Exception as e:
    raise Exception(f"Error loading files: {e}")

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the form
        age = int(request.form['age'])
        weight = float(request.form['weight'])
        stress_level = request.form['stress_level']
        sleep_duration = float(request.form['sleep_duration'])
        exercise_habit = request.form['exercise_habit']
        activity_level = request.form['activity_level']
        medical_history = request.form['medical_history']

        # Default values for optional inputs
        height = 170  # Default height in cm
        heart_rate = 75  # Default heart rate in bpm
        caffeine_intake = 'None'  # Default caffeine intake
        study_hours = 4  # Default study hours

        # Prepare the input DataFrame
        sample_input = pd.DataFrame({
            'Age': [age],
            'Weight': [weight],
            'Height': [height],
            'Heart_Rate': [heart_rate],
            'Activity_Level': [label_encoders['Activity_Level'].transform([activity_level])[0]],
            'Medical_History': [label_encoders['Medical_History'].transform([medical_history])[0]],
            'Stress_Level': [label_encoders['Stress_Level'].transform([stress_level])[0]],
            'Sleep_Duration': [sleep_duration],
            'Exercise_Habit': [label_encoders['Exercise_Habit'].transform([exercise_habit])[0]],
            'Caffeine_Intake': [label_encoders['Caffeine_Intake'].transform([caffeine_intake])[0]],
            'Study_Hours': [study_hours]
        })

        # Ensure the columns match the training data
        sample_input = sample_input[model.feature_names_in_]

        # Normalize the numerical features
        numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
        sample_input[numerical_columns] = scaler.transform(sample_input[numerical_columns])

        # Make the prediction
        prediction = model.predict(sample_input)
        predicted_risk = label_encoders['Risk_Level'].inverse_transform(prediction)[0]

        # Redirect to the result page with the predicted risk level
        return render_template('result.html', risk_level=predicted_risk)

    except Exception as e:
        return jsonify({'error': str(e)})

# Define the result page route
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
