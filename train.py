import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("StudentPerformanceFactors.csv")

X = df[
    [
        'Hours_Studied',
        'Attendance',
        'Sleep_Hours',
        'Previous_Scores',
        'Tutoring_Sessions'
    ]
]

y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor()

model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')

print("Model Trained Successfully")