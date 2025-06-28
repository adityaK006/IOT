# import joblib
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix

# # Load the model, scaler, and label encoders
# try:
#     model = joblib.load('IOT project/student_health_risk_model2.pkl')
#     scaler = joblib.load('IOT project/scaler2.pkl')
#     label_encoders = joblib.load('IOT project/label_encoders2.pkl')
#     print("Model, scaler, and encoders loaded successfully.")
# except Exception as e:
#     raise Exception(f"Error loading files: {e}")

# # Load your test dataset
# try:
#     # Replace 'test_data.csv' with your actual test dataset path
#     test_data = pd.read_csv('IOT project/balanced_student_health_dataset.csv')
#     print("Test dataset loaded successfully.")
# except Exception as e:
#     raise Exception(f"Error loading test dataset: {e}")

# # Separate features and labels
# try:
#     X_test = test_data.drop(columns=['Risk_Level'])  # Replace 'Risk_Level' with your target column name
#     y_test = test_data['Risk_Level']
# except Exception as e:
#     raise Exception(f"Error splitting features and labels: {e}")

# # Preprocess the test data
# try:
#     # Transform categorical features using label encoders
#     for col in label_encoders.keys():
#         if col in X_test.columns:
#             X_test[col] = label_encoders[col].transform(X_test[col])

#     # Normalize numerical features
#     numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
#     X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

#     # Ensure columns match the model's input features
#     X_test = X_test[model.feature_names_in_]
# except Exception as e:
#     raise Exception(f"Error during preprocessing: {e}")

# # Test the model and generate the report
# try:
#     y_pred = model.predict(X_test)
#     y_pred_decoded = label_encoders['Risk_Level'].inverse_transform(y_pred)

#     print("Model Accuracy Report:")
#     print(classification_report(y_test, y_pred_decoded))

#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred_decoded))
# except Exception as e:
#     raise Exception(f"Error during model evaluation: {e}")





# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix

# # Load the model, scaler, and label encoders
# try:
#     model = joblib.load('student_health_risk_model2.pkl')
#     scaler = joblib.load('scaler2.pkl')
#     label_encoders = joblib.load('label_encoders2.pkl')
#     print("Model, scaler, and encoders loaded successfully.")
# except Exception as e:
#     raise Exception(f"Error loading files: {e}")

# # Generate synthetic random data
# def generate_synthetic_data(num_samples=100):
#     np.random.seed(42)  # For reproducibility

#     # Define ranges or categories for each feature
#     data = {
#         'Age': np.random.randint(18, 35, size=num_samples),  # Age between 18 and 35
#         'Weight': np.random.uniform(45, 100, size=num_samples),  # Weight in kg
#         'Height': np.random.uniform(150, 200, size=num_samples),  # Height in cm
#         'Heart_Rate': np.random.uniform(60, 120, size=num_samples),  # Heart rate in bpm
#         'Activity_Level': np.random.choice(label_encoders['Activity_Level'].classes_, size=num_samples),
#         'Medical_History': np.random.choice(label_encoders['Medical_History'].classes_, size=num_samples),
#         'Stress_Level': np.random.choice(label_encoders['Stress_Level'].classes_, size=num_samples),
#         'Sleep_Duration': np.random.uniform(3, 10, size=num_samples),  # Sleep duration in hours
#         'Exercise_Habit': np.random.choice(label_encoders['Exercise_Habit'].classes_, size=num_samples),
#         'Caffeine_Intake': np.random.choice(label_encoders['Caffeine_Intake'].classes_, size=num_samples),
#         'Study_Hours': np.random.uniform(1, 10, size=num_samples)  # Study hours per day
#     }

#     # Create a DataFrame
#     synthetic_data = pd.DataFrame(data)
#     return synthetic_data

# # Generate synthetic data
# synthetic_data = generate_synthetic_data(num_samples=100)

# # Preprocess the synthetic data
# try:
#     # Transform categorical features using label encoders
#     for col in label_encoders.keys():
#         if col in synthetic_data.columns:
#             synthetic_data[col] = label_encoders[col].transform(synthetic_data[col])

#     # Normalize numerical features
#     numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
#     synthetic_data[numerical_columns] = scaler.transform(synthetic_data[numerical_columns])

#     # Ensure columns match the model's input features
#     synthetic_data = synthetic_data[model.feature_names_in_]
# except Exception as e:
#     raise Exception(f"Error during preprocessing: {e}")

# # Test the model on synthetic data
# try:
#     predictions = model.predict(synthetic_data)
#     predicted_labels = label_encoders['Risk_Level'].inverse_transform(predictions)

#     print("Generated Synthetic Data (First 5 rows):")
#     print(synthetic_data.head())

#     print("\nModel Predictions on Synthetic Data:")
#     print(predicted_labels[:5])  # Show the first 5 predictions

#     # Since this is synthetic data, evaluate classification distribution
#     unique, counts = np.unique(predicted_labels, return_counts=True)
#     print("\nPrediction Distribution:")
#     for label, count in zip(unique, counts):
#         print(f"{label}: {count}")

# except Exception as e:
#     raise Exception(f"Error during model evaluation: {e}")






#accuracy 24%
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# # Load the model, scaler, and label encoders
# try:
#     model = joblib.load('student_health_risk_model2.pkl')
#     scaler = joblib.load('scaler2.pkl')
#     label_encoders = joblib.load('label_encoders2.pkl')
#     print("Model, scaler, and encoders loaded successfully.")
# except Exception as e:
#     raise Exception(f"Error loading files: {e}")

# # Generate synthetic random data
# def generate_synthetic_data(num_samples=100):
#     np.random.seed(42)  # For reproducibility

#     # Define ranges or categories for each feature
#     data = {
#         'Age': np.random.randint(18, 35, size=num_samples),  # Age between 18 and 35
#         'Weight': np.random.uniform(45, 100, size=num_samples),  # Weight in kg
#         'Height': np.random.uniform(150, 200, size=num_samples),  # Height in cm
#         'Heart_Rate': np.random.uniform(60, 120, size=num_samples),  # Heart rate in bpm
#         'Activity_Level': np.random.choice(label_encoders['Activity_Level'].classes_, size=num_samples),
#         'Medical_History': np.random.choice(label_encoders['Medical_History'].classes_, size=num_samples),
#         'Stress_Level': np.random.choice(label_encoders['Stress_Level'].classes_, size=num_samples),
#         'Sleep_Duration': np.random.uniform(3, 10, size=num_samples),  # Sleep duration in hours
#         'Exercise_Habit': np.random.choice(label_encoders['Exercise_Habit'].classes_, size=num_samples),
#         'Caffeine_Intake': np.random.choice(label_encoders['Caffeine_Intake'].classes_, size=num_samples),
#         'Study_Hours': np.random.uniform(1, 10, size=num_samples)  # Study hours per day
#     }

#     # Create a DataFrame
#     synthetic_data = pd.DataFrame(data)
#     return synthetic_data

# # Generate synthetic data
# synthetic_data = generate_synthetic_data(num_samples=100)

# # Preprocess the synthetic data
# try:
#     # Transform categorical features using label encoders
#     for col in label_encoders.keys():
#         if col in synthetic_data.columns:
#             synthetic_data[col] = label_encoders[col].transform(synthetic_data[col])

#     # Normalize numerical features
#     numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
#     synthetic_data[numerical_columns] = scaler.transform(synthetic_data[numerical_columns])

#     # Ensure columns match the model's input features
#     synthetic_data = synthetic_data[model.feature_names_in_]
# except Exception as e:
#     raise Exception(f"Error during preprocessing: {e}")

# # Define pseudo-label generation logic
# def generate_pseudo_labels(data):
#     # High Risk: High stress, very low sleep, rare exercise, or high weight
#     high_risk = (
#         (data['Stress_Level'] > 2) |  # High stress
#         (data['Sleep_Duration'] < 5) |  # Very low sleep
#         (data['Exercise_Habit'] == 0) |  # Rare exercise (example encoded value)
#         (data['Weight'] > 80)  # Reasonably high weight
#     )

#     # Low Risk: Good sleep, low stress, regular exercise
#     low_risk = (
#         (data['Stress_Level'] < 2) &
#         (data['Sleep_Duration'] > 7) &
#         (data['Exercise_Habit'] == 1)  # Regular exercise
#     )

#     # Moderate Risk: All others
#     moderate_risk = ~(high_risk | low_risk)

#     # Assign pseudo-labels
#     risk_levels = ['High', 'Low', 'Moderate']
#     pseudo_labels = pd.Series(
#         np.select([high_risk, low_risk, moderate_risk], risk_levels, default='Moderate'),
#         index=data.index
#     )
#     return pseudo_labels

# # Generate pseudo-labels
# pseudo_labels = generate_pseudo_labels(synthetic_data)

# # Test the model on synthetic data
# try:
#     predictions = model.predict(synthetic_data)
#     predicted_labels = label_encoders['Risk_Level'].inverse_transform(predictions)

#     # Evaluate the model's predictions against pseudo-labels
#     accuracy = accuracy_score(pseudo_labels, predicted_labels)
#     print(f"\nAccuracy of the model on synthetic data: {accuracy:.2f}")

#     # Classification report
#     print("\nClassification Report:")
#     print(classification_report(pseudo_labels, predicted_labels))

#     # Confusion matrix
#     print("\nConfusion Matrix:")
#     print(pd.DataFrame(
#         confusion_matrix(pseudo_labels, predicted_labels),
#         index=['High', 'Low', 'Moderate'],
#         columns=['High', 'Low', 'Moderate']
#     ))

#     # Distribution of predictions
#     unique, counts = np.unique(predicted_labels, return_counts=True)
#     print("\nPrediction Distribution:")
#     for label, count in zip(unique, counts):
#         print(f"{label}: {count}")

# except Exception as e:
#     raise Exception(f"Error during model evaluation: {e}")


# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics import classification_report, confusion_matrix

# # Load the model, scaler, and label encoders
# try:
#     model = joblib.load('student_health_risk_model2.pkl')
#     scaler = joblib.load('scaler2.pkl')
#     label_encoders = joblib.load('label_encoders2.pkl')
#     print("Model, scaler, and encoders loaded successfully.")
# except Exception as e:
#     raise Exception(f"Error loading files: {e}")

# # Load the dataset
# file_path = r"C:\Users\ADITYA\Downloads\difficult_test_dataset_fixed.csv"
# try:
#     test_data = pd.read_csv(file_path)
#     print("Dataset loaded successfully.")
# except Exception as e:
#     raise Exception(f"Error loading dataset: {e}")

# # Handle unseen labels and preprocess the data
# try:
#     # Map unseen labels to 'Unknown' and update label encoders
#     for col in label_encoders.keys():
#         if col in test_data.columns:
#             # Identify unseen labels
#             unseen_labels = set(test_data[col].unique()) - set(label_encoders[col].classes_)
#             if unseen_labels:
#                 print(f"Unseen labels found in column '{col}': {unseen_labels}")
#                 # Add 'Unknown' category to the encoder
#                 label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'Unknown')
#                 # Replace unseen labels with 'Unknown'
#                 test_data[col] = test_data[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'Unknown')
#             # Transform the column
#             test_data[col] = label_encoders[col].transform(test_data[col])

#     # Normalize numerical features
#     numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
#     test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

#     # Ensure the columns match the model's input features
#     test_data = test_data[model.feature_names_in_]
#     print("Preprocessing completed successfully.")
# except Exception as e:
#     raise Exception(f"Error during preprocessing: {e}")

# # Test the model on the processed dataset
# try:
#     predictions = model.predict(test_data)
#     predicted_labels = label_encoders['Risk_Level'].inverse_transform(predictions)

#     # Display classification report
#     print("\nClassification Report:")
#     print(classification_report(test_data['Risk_Level'], predicted_labels))

#     # Display confusion matrix
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(test_data['Risk_Level'], predicted_labels))

# except Exception as e:
#     raise Exception(f"Error during model evaluation: {e}")








# import joblib
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load the model, scaler, and label encoders
# try:
#     model = joblib.load('E:\COLLEGE\IOT\IOT project\student_health_risk_model2.pkl')
#     scaler = joblib.load('E:\COLLEGE\IOT\IOT project\scaler2.pkl')
#     label_encoders = joblib.load('E:\COLLEGE\IOT\IOT project\label_encoders2.pkl')
#     print("Model, scaler, and encoders loaded successfully.")
# except Exception as e:
#     raise Exception(f"Error loading files: {e}")

# # Load your full dataset
# try:
#     # Replace 'dataset.csv' with your actual dataset path
#     data = pd.read_csv('E:\COLLEGE\IOT\IOT project\balanced_student_health_dataset.csv')
#     print("Dataset loaded successfully.")
# except Exception as e:
#     raise Exception(f"Error loading dataset: {e}")

# # Separate features and labels
# try:
#     X = data.drop(columns=['Risk_Level'])  # Features
#     y = data['Risk_Level']  # Target variable
# except Exception as e:
#     raise Exception(f"Error splitting features and labels: {e}")

# # Step 1: Split the data into training and testing sets (80-20 split)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Preprocess the data
# try:
#     # Transform categorical features using label encoders (now with combined data for both train and test)
#     for col in label_encoders.keys():
#         if col in X.columns:
#             # Combine training and testing data for categorical column and fit the encoder
#             combined_data = pd.concat([X_train[col], X_test[col]], axis=0)
#             label_encoders[col].fit(combined_data)  # Fit on combined data
#             X_train[col] = label_encoders[col].transform(X_train[col])
#             X_test[col] = label_encoders[col].transform(X_test[col])

#     # Normalize numerical features
#     numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
#     X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
#     X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# except Exception as e:
#     raise Exception(f"Error during preprocessing: {e}")

# # Step 3: Test the model on the test set
# try:
#     y_pred = model.predict(X_test)
#     y_pred_decoded = label_encoders['Risk_Level'].inverse_transform(y_pred)

#     print("Model Accuracy Report:")
#     print(classification_report(y_test, y_pred_decoded))

#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred_decoded))
# except Exception as e:
#     raise Exception(f"Error during model evaluation: {e}")






#updated 4th march 2025
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the model, scaler, and label encoders
try:
    model = joblib.load(r'E:\COLLEGE\IOT\IOT project\student_health_risk_model2.pkl')
    scaler = joblib.load(r'E:\COLLEGE\IOT\IOT project\scaler2.pkl')
    label_encoders = joblib.load(r'E:\COLLEGE\IOT\IOT project\label_encoders2.pkl')
    print("Model, scaler, and encoders loaded successfully.")
except Exception as e:
    raise Exception(f"Error loading files: {e}")

# Load your full dataset
try:
    # Use raw string (r'path') or double backslashes
    data = pd.read_csv(r'E:\COLLEGE\IOT\IOT project\balanced_student_health_dataset.csv')
    print("Dataset loaded successfully.")
except Exception as e:
    raise Exception(f"Error loading dataset: {e}")

# Separate features and labels
try:
    X = data.drop(columns=['Risk_Level'])  # Features
    y = data['Risk_Level']  # Target variable
except Exception as e:
    raise Exception(f"Error splitting features and labels: {e}")

# Step 1: Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
try:
    # Transform categorical features using label encoders
    for col in label_encoders.keys():
        if col in X.columns:
            combined_data = pd.concat([X_train[col], X_test[col]], axis=0)
            label_encoders[col].fit(combined_data)  # Fit on combined data
            X_train[col] = label_encoders[col].transform(X_train[col])
            X_test[col] = label_encoders[col].transform(X_test[col])

    # Normalize numerical features
    numerical_columns = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
    X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

except Exception as e:
    raise Exception(f"Error during preprocessing: {e}")

# Step 3: Test the model on the test set
try:
    y_pred = model.predict(X_test)
    y_pred_decoded = label_encoders['Risk_Level'].inverse_transform(y_pred)

    print("Model Accuracy Report:")
    print(classification_report(y_test, y_pred_decoded))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_decoded))
except Exception as e:
    raise Exception(f"Error during model evaluation: {e}")
