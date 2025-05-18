import joblib
import pandas as pd
import streamlit as st


models = joblib.load('heart_models.pkl')


st.title("Heart Disease Prediction Models")

# Input sections
st.header("User Details")
col1, col2 = st.columns(2)
with col1:
    
 st.write('Predict Heart Disease:')
age = st.number_input('Age')
sex = st.selectbox('Sex', [0, 1])
cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure')
chol = st.number_input('Cholesterol')
fbs = st.selectbox('Fasting Blood Sugar', [0, 1])
restecg = st.selectbox('Resting Electrocardiogram', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate')
exang = st.selectbox('Exercise-Induced Angina', [0, 1])
oldpeak = st.number_input('ST Depression')
slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2])
ca = st.number_input('Number of Major Vessels')
thal = st.selectbox('Thalassemia', [0, 1, 2, 3])
    

# Model Selection
st.header("Model Selection")
selected_model = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"])
if st.button("Predict"):
    user_data = pd.DataFrameinput_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})
    if selected_model == "Logistic Regression":
        prediction = models['logistic_regression'].predict(user_data)
    elif selected_model == "Decision Tree":
        prediction = models['decision_tree'].predict(user_data)
    elif selected_model == "Random Forest":
        prediction = models['random_forest'].predict(user_data)
    else:
            prediction = models['XGBoost'].predict(user_data)
    st.write(f"{selected_model} Heart Prediction:", prediction)
    
    
    # Display prediction
st.write(f'Prediction using {selected_model}:')
if prediction[0] == 1:
    st.write('Heart disease predicted')
else:
    st.write('No heart disease predicted')
    
st.markdown("---")
st.markdown("<center>Made by RedCherry</center>", unsafe_allow_html=True)