import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Loading the model file
model = joblib.load("C:/Users/dell/.spyder-py3/titanic_logistic_model.pkl")

# Setting a nice title for the app, for users
st.title("Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival probability.")

# Getting user inputs, keeping it simple with options they can choose
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.slider("Age", 0, 100, 30)  # Default to 30, seems reasonable for most passengers
sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.slider("Fare", 0.0, 500.0, 32.0)  # Setting a practical fare range
sex = st.selectbox("Sex", ["female", "male"])
embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

# Preparing the input data to match what the model was trained on
input_data = {
    "PassengerId": 1,  # Just a dummy value, not really used in prediction
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Sex": sex,
    "Embarked": embarked
}

# Converting to a DataFrame, like organizing a passenger list
input_df = pd.DataFrame([input_data])

# Encoding categorical variables to match training data format
input_df = pd.get_dummies(input_df, columns=["Sex", "Embarked"], drop_first=True)

# Ensuring all expected columns are present
training_columns = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Re-ordering columns to align with training data structure
input_df = input_df[training_columns]

# Making the prediction when the button is clicked
if st.button("Predict"):
    probability = model.predict_proba(input_df)[:, 1][0]
    prediction = "Survived" if probability >= 0.5 else "Did Not Survive"
    st.write(f"**Survival Probability**: {probability:.2%}")
    st.write(f"**Prediction**: {prediction}")