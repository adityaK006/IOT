import pandas as pd

# Define the data (same as you provided)
data = {
    'Age': [18, 22, 30],
    'Weight': [50, 75, 85],
    'Height': [170, 175, 180],
    'Heart_Rate': [72, 85, 65],
    'Activity_Level': ['Resting', 'Walking', 'Running'],
    'Medical_History': ['No History', 'Diabetes', 'Cardiac Risk'],
    'Stress_Level': ['Low', 'High', 'Medium'],
    'Sleep_Duration': [8, 5, 6],
    'Exercise_Habit': ['Regular', 'Occasional', 'Rarely'],
    'Caffeine_Intake': ['None', 'None', 'None'],
    'Study_Hours': [5, 4, 6]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = 'E:/COLLEGE/IOT/IOT project/unseen_test_data.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file successfully created at: {csv_file_path}")
