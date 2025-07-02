import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import streamlit as st

model = tf.keras.models.load_model("regressionModel.h5")

with open('gender_label_encoder.pkl','rb') as file:
    gender_encoder = pickle.load(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    geo_encoder = pickle.load(file)
with open('churn_scaler.pkl','rb') as file:
    scaler = pickle.load(file)

    st.title("Salary Prediction")
    geography = st.selectbox('Geography' , geo_encoder.categories_[0])
    gender = st.selectbox('Gender' , gender_encoder.classes_)
    age = st.slider('Age',18,92,25)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    tenure = st.slider('Tenure' , 0,10)
    num_of_products =st.slider('Number of Produts',1,4)
    has_cr_card = st.selectbox('has Credit Card',[0,1])
    exited = st.selectbox("Exited" , [0,1])
    is_active_member = st.selectbox('Is Active Member' , [0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : gender_encoder.transform([gender]),
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' :[has_cr_card],
    'IsActiveMember' : [is_active_member],
    'Exited' : [exited],
})

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded , columns = geo_encoder.get_feature_names_out(['Geography']))

input_df = pd.concat([input_data.reset_index(drop=True) , geo_encoded_df],axis=1)
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)

st.sidebar.write(f"The estimated Salary of the person is : {prediction[0][0]:.0f}")