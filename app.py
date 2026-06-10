import streamlit as st
import joblib

model = joblib.load('model.pkl')

st.title("Student Performance Predictor")

hours = st.number_input("Hours Studied")
attendance = st.number_input("Attendance")
sleep = st.number_input("Sleep Hours")
previous = st.number_input("Previous Score")
tutoring = st.number_input("Tutoring Sessions")

if st.button("Predict"):
    prediction = model.predict(
        [[hours, attendance, sleep, previous, tutoring]]
    )

    st.success(
        f"Predicted Exam Score: {prediction[0]:.2f}"
    )