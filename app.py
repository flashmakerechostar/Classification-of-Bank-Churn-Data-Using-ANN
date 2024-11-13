import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle
import streamlit as st

# load saved model
model = tf.keras.models.load_model('model.keras')

# load the pickle files

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender_loaded = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo_loaded = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)
    
# steamlit App

st.title("Customer Churn Prediction")

# user input

credit_score = st.number_input('Credit Score')
gender = st.selectbox("Gender",label_encoder_gender_loaded.classes_ )
age = st.slider("Age", 18,92)
tenure = st.slider('Tenure', 0,10)
balance = st.number_input("Balance")
num_of_products = st.slider("Number of Products", 1,4)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Memeber", [0,1])
estimated_salary = st.number_input("Estimated Salary")
geography = st.selectbox('Geography', onehot_encoder_geo_loaded.categories_[0])

# replicating orginal train dataframe

input_data = pd.DataFrame({'CreditScore' : [credit_score],
 'Gender':label_encoder_gender_loaded.transform([gender])[0],
 'Age':[age],
 'Tenure':[tenure],
 'Balance':[balance],
 'NumOfProducts':[num_of_products],
 'HasCrCard':[has_cr_card],
 'IsActiveMember':[is_active_member],
 'EstimatedSalary':[estimated_salary],
 })
 
 
geo_transformed = onehot_encoder_geo_loaded.transform([[geography]])
geo_df = pd.DataFrame(geo_transformed, columns=['Geography_France', 'Geography_Germany', 'Geography_Spain'])
input_data = pd.concat((input_data.reset_index(drop=True), geo_df), axis=1)
input_data_scaled = scaler_loaded.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]
st.write(f'Churn Probability : {prediction_probability : .2f}')
if prediction_probability > 0.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is not likely to churn")