# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the model, scaler, and label encoders
# model = joblib.load('student_health_risk_model2.pkl')
# scaler = joblib.load('scaler2.pkl')
# label_encoders = joblib.load('label_encoders2.pkl')

# # Load your dataset again
# df = pd.read_csv(r'E:\COLLEGE\IOT\IOT project\balanced_student_health_dataset.csv')


# # Encode categorical features (same as training)
# categorical_cols = ['Activity_Level', 'Medical_History', 'Stress_Level', 'Exercise_Habit', 'Caffeine_Intake', 'Risk_Level']
# for col in categorical_cols:
#     df[col] = label_encoders[col].transform(df[col])

# # Split features and target
# X = df.drop(columns=['Risk_Level'])
# y = df['Risk_Level']

# # Scale the numerical data (same as training)
# X[['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']] = scaler.transform(
#     X[['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']]
# )

# # Predict on the same dataset (since you don't have separate test data)
# y_pred = model.predict(X)

# # -------------------------------
# # ✅ 1. Accuracy Score
# # -------------------------------
# accuracy = accuracy_score(y, y_pred)
# print("Model Accuracy: {:.2f}%".format(accuracy * 100))

# # -------------------------------
# # ✅ 2. Classification Report (Precision, Recall, F1-score)
# # -------------------------------
# print("\nClassification Report:\n")
# print(classification_report(y, y_pred))

# # -------------------------------
# # ✅ 3. Confusion Matrix
# # -------------------------------
# conf_matrix = confusion_matrix(y, y_pred)

# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=label_encoders['Risk_Level'].classes_,
#             yticklabels=label_encoders['Risk_Level'].classes_)
# plt.title('Confusion Matrix (Actual vs Predicted)')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # -------------------------------
# # ✅ 4. ROC-AUC Score
# # -------------------------------
# # Convert y to binary for ROC-AUC (you have multi-class)
# y_bin = pd.get_dummies(y)  # One hot encoding for ROC
# y_pred_proba = model.predict_proba(X)

# # Plot ROC Curve
# plt.figure(figsize=(8, 6))
# for i, class_name in enumerate(label_encoders['Risk_Level'].classes_):
#     fpr, tpr, _ = roc_curve(y_bin.iloc[:, i], y_pred_proba[:, i])
#     auc_score = roc_auc_score(y_bin.iloc[:, i], y_pred_proba[:, i])
#     plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.2f})')

# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC-AUC Curve for Random Forest Classifier')
# plt.legend(loc='lower right')
# plt.show()

# # -------------------------------
# # ✅ 5. ROC-AUC Score for Entire Model
# # -------------------------------
# macro_auc = roc_auc_score(y_bin, y_pred_proba, average='macro')
# print("Macro-Averaged ROC-AUC Score: {:.2f}".format(macro_auc))






import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

# Load the trained model, scaler, and encoders
model = joblib.load('student_health_risk_model2.pkl')
scaler = joblib.load('scaler2.pkl')
label_encoders = joblib.load('label_encoders2.pkl')

# ✅ Step 1: Generate 1000 Random Test Samples
np.random.seed(42)
age = np.random.randint(18, 60, 1000)
weight = np.random.randint(45, 100, 1000)
height = np.random.randint(150, 190, 1000)
heart_rate = np.random.randint(60, 100, 1000)
activity_level = np.random.choice(['Resting', 'Walking', 'Running'], 1000)
medical_history = np.random.choice(['No History', 'Diabetes', 'Cardiac Risk'], 1000)
stress_level = np.random.choice(['Low', 'Medium', 'High'], 1000)
sleep_duration = np.random.uniform(4, 9, 1000)
exercise_habit = np.random.choice(['Regular', 'Occasional', 'Rarely'], 1000)
caffeine_intake = np.random.choice(['None', '1 Cup', '2 Cups', '>2 Cups'], 1000)
study_hours = np.random.randint(1, 12, 1000)

# ✅ Step 2: Create a DataFrame
test_data = pd.DataFrame({
    'Age': age,
    'Weight': weight,
    'Height': height,
    'Heart_Rate': heart_rate,
    'Activity_Level': label_encoders['Activity_Level'].transform(activity_level),
    'Medical_History': label_encoders['Medical_History'].transform(medical_history),
    'Stress_Level': label_encoders['Stress_Level'].transform(stress_level),
    'Sleep_Duration': sleep_duration,
    'Exercise_Habit': label_encoders['Exercise_Habit'].transform(exercise_habit),
    'Caffeine_Intake': label_encoders['Caffeine_Intake'].transform(caffeine_intake),
    'Study_Hours': study_hours
})

# ✅ Step 3: Normalize the Data
numerical_cols = ['Age', 'Weight', 'Height', 'Heart_Rate', 'Sleep_Duration', 'Study_Hours']
test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])

# ✅ Step 4: Force Exact Feature Order (VERY IMPORTANT)
# This is the feature order the model was trained on
expected_columns = model.feature_names_in_

# ✅ Drop any unexpected columns (like Predicted_Risk_Level if present)
test_data = test_data[expected_columns]

# ✅ Step 5: Predict Risk Level
predictions = model.predict(test_data)
predicted_risk = label_encoders['Risk_Level'].inverse_transform(predictions)

# ✅ Step 6: Append Predictions to DataFrame
test_data['Predicted_Risk_Level'] = predicted_risk

# ✅ Step 7: Save Predictions to CSV
output_file = 'E:/COLLEGE/IOT/IOT project/prediction_1000_samples.csv'
test_data.to_csv(output_file, index=False)
print(f"\n✅ Predictions saved successfully to: {output_file}")

# ✅ Step 8: Classification Report
print("\n✅ Classification Report:\n")
print(classification_report(predictions, label_encoders['Risk_Level'].transform(predicted_risk)))

# ✅ Step 9: Confusion Matrix
cm = confusion_matrix(predictions, label_encoders['Risk_Level'].transform(predicted_risk))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Moderate', 'High'], yticklabels=['Low', 'Moderate', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ✅ Step 10: Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': test_data.drop(columns=['Predicted_Risk_Level']).columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

print("\n✅ Code executed successfully! 🚀")
