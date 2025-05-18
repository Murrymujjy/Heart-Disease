import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset

data = pd.read_csv("heart.csv")
# Assuming you have a DataFrame 'df' with the dataset

# Define features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier()
}

# Train and evaluate models
model_accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy

# Streamlit app
st.title('Heart Disease Prediction Models')

# Model performance
st.write('Model Accuracies:')
for name, accuracy in model_accuracies.items():
    st.write(f'{name}: {accuracy:.3f}')

# Prediction
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

# Create a DataFrame for prediction
input_data = pd.DataFrame({
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

# Make prediction
selected_model = st.selectbox('Select Model', list(models.keys()))
model = models[selected_model]
prediction = model.predict(input_data)

# Display prediction
st.write(f'Prediction using {selected_model}:')
if prediction[0] == 1:
    st.write('Heart disease predicted')
else:
    st.write('No heart disease predicted')
    
    
st.markdown("---")
st.markdown("<center>Made by RedCherry</center>", unsafe_allow_html=True)