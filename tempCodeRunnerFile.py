import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load the model, scaler, and label encoders
try:
    model = joblib.load('IOT project/student_health_risk_model2.pkl')
    scaler = joblib.load('IOT project/scaler2.pkl')
    label_encoders = joblib.load('IOT project/label_encoders2.pkl')
    print("Model, scaler, and encoders loaded successfully.")
except Exception as e:
    raise Exception(f"Error loading files: {e}")

# Load your test dataset
try:
    # Replace 'test_data.csv' with your actual test dataset path
    test_data = pd.read_csv('IOT project/balanced_student_health_dataset.csv')
    print("Test dataset loaded successfully.")
except Exception as e:
    raise Exception(f"Error loading test dataset: {e}")

# Separate features and labels
try:
    X_test = test_data.drop(columns=['Risk_Level'])  # Replace 'Risk_Level' with your target column name
    y_test = test_data['Risk_Level']
except Exception as e:
    raise Exception(f"Error splitting features and labels: {e}")

# Preprocess the test data
try:
    # Transform categorical features using label encoders
    for col in label_encoders.keys():
        if col in X_test.columns:
            X_test[col] = label_encoders[col].transform(X_test[col])

    # Normalize numerical features
    numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Ensure columns match the model's input features
    X_test = X_test[model.feature_names_in_]
except Exception as e:
    raise Exception(f"Error during preprocessing: {e}")

# Test the model and generate the report
try:
    y_pred = model.predict(X_test)
    y_pred_decoded = label_encoders['Risk_Level'].inverse_transform(y_pred)

    print("Model Accuracy Report:")
    print(classification_report(y_test, y_pred_decoded))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_decoded))
except Exception as e:
    raise Exception(f"Error during model evaluation: {e}")