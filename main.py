import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_students = 200
data = {
    'GPA': np.random.uniform(0, 4, num_students),
    'Attendance': np.random.randint(0, 100, num_students), # Percentage
    'Income': np.random.randint(20000, 100000, num_students),
    'Parent_Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], num_students),
    'At_Risk': np.random.choice([0, 1], num_students, p=[0.7, 0.3]) # 30% at risk
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Parent_Education'], drop_first=True)
# --- 3. Data Splitting ---
X = df.drop('At_Risk', axis=1)
y = df['At_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 4. Model Training ---
model = LogisticRegression()
model.fit(X_train, y_train)
# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# --- 6. Visualization ---
# Example: GPA vs. Attendance colored by At_Risk status
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GPA', y='Attendance', hue='At_Risk', data=df, palette='viridis')
plt.title('GPA vs. Attendance colored by At-Risk Status')
plt.xlabel('GPA')
plt.ylabel('Attendance (%)')
plt.savefig('gpa_attendance_scatter.png')
print("Plot saved to gpa_attendance_scatter.png")
# Example:  Distribution of Income for At-Risk and Not At-Risk students.
plt.figure(figsize=(8,6))
sns.histplot(data=df, x='Income', hue='At_Risk', kde=True, element="step")
plt.title('Income Distribution by At-Risk Status')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.savefig('income_distribution.png')
print("Plot saved to income_distribution.png")