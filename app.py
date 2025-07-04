import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
import streamlit as st
import pickle

model = tf.keras.models.load_model('model.h5')

with open('gender_label_encoder.pkl','rb') as file:
    gender_label_encoder = pickle.load(file)
with open('churn_modelling_scaler.pkl','rb') as file:
    scaler = pickle.load(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
#Streamlit App
st.title("Customer Churn Prediction")
#inputs : 
geography = st.selectbox('Geography' , onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender' , gender_label_encoder.classes_)
age = st.slider('Age',18,92,25)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure' , 0,10)
num_of_products =st.slider('Number of Produts',1,4)
has_cr_card = st.selectbox('has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member' , [0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [gender_label_encoder.transform([gender])[0]],
    'Age' : [age],
    'Tenure':[tenure],
    'Balance' : [balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded , columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_data.reset_index(drop=True) , geo_encoded_df],axis=1)
#scaling the data
input_scaled = scaler.transform(input_df)
#predicting the output
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]
if prediction_proba>0.5:
    st.write("The customer is likely to churn.")
    st.write(f"Churn Probability :  {prediction_proba:.2f}")
else:
    st.write("The customer is not likely to churn.")
    st.write(f"Churn Probability :  {prediction_proba:.2f}")