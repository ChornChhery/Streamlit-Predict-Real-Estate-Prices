import streamlit as st
import numpy as np
import joblib


scaler = joblib.load('scaler.pkl')

model = joblib.load('model.pkl')


st.title("Real Estate Price Prediction App")


st.divider()

bed = st.number_input("Enter the number of bedrooms", value=2, step = 1)
bath = st.number_input("Enter the number of bathrooms", value=1, step = 1)
size = st.number_input("Enter the size of the house in sqft", value=1000, step = 50)


X = [bed,bath,size]

st.divider()


y_pred = st.button("Predict!")


st.divider()

if y_pred:

    st.balloons()

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]


    st.write(f"The predicted price of the house is ${prediction:,.2f}")

else:
    st.write("Please enter the number of bedrooms, bathrooms, and size of the house to predict the price.")