import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the model and encoders
model = load_model('model.keras')

with open('scaler.pkl','rb') as f:
  sc = pickle.load(f)

with open('encoder_gender.pkl','rb') as f:
  le = pickle.load(f)

with open('ohe_geo.pkl','rb') as f:
  ohe = pickle.load(f)


st.title('Customer Churn Prediction')

st.header('Enter Customer Details:')

credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure (Years)', min_value=0, max_value=10)
balance = st.number_input('Account Balance', value=100000.0)
num_products = st.slider('Number of Products', min_value=1, max_value=4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
estimated_salary = st.number_input('Estimated Salary')


if st.button('Predict Churn'):
    input_data ={
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts':num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
    }

    input_df = pd.DataFrame(input_data,index=[0])

    # Encode categorical variables
    input_df['Gender'] = le.transform(input_df['Gender'])
    g_enc = ohe.transform([[input_df['Geography'][0]]]).toarray()
    g_enc_df = pd.DataFrame(g_enc,columns = ohe.get_feature_names_out(['Geography']))

    input_df = pd.concat([input_df.drop('Geography',axis = 1),g_enc_df],axis = 1)

    # Scale the data
    scaled_data = sc.transform(input_df)

    # Make prediction
    pred = model.predict(scaled_data)
    pred_prob = pred[0][0]

    st.subheader('Prediction Result:')
    if pred_prob > 0.5:
        st.error(f'The customer is likely to exit the bank with a probability of {pred_prob:.2f}')
    else:
        st.success(f'The customer is likely to stay with the bank with a probability of {pred_prob:.2f}')
