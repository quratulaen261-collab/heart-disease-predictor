import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.title("🩺 Health Disease Predictor")
st.write("Select a disease, enter patient info, and get a prediction.")

disease = st.selectbox("Choose Disease", ["Heart Disease"])  

if disease == "Heart Disease":
    # User input fields
    age = st.number_input("Age", 0, 120, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    rest_bp = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fasting_bs = st.number_input("Fasting Blood Sugar (0 or 1)", 0, 1, 0)
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Load and prepare dataset
    df = pd.read_csv("C:\\Users\\SAKHAWAT\\Downloads\\archive\\heart.csv")
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # Create input DataFrame for prediction
    input_data = pd.DataFrame({
        'Age':[age], 'Sex':[sex], 'ChestPainType':[chest_pain], 'RestingBP':[rest_bp],
        'Cholesterol':[chol], 'FastingBS':[fasting_bs], 'RestingECG':[rest_ecg],
        'MaxHR':[max_hr], 'ExerciseAngina':[exercise_angina], 'Oldpeak':[oldpeak], 'ST_Slope':[st_slope]
    })

    # One-hot encode input like training data
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Predict on button click
    if st.button("Predict"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("Prediction: Heart Disease Detected!")
        else:
            st.success("Prediction: No Heart Disease Detected.")








