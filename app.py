import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction App")
st.write("This ML model predicts house price based on area and number of bedrooms.")
data = {
    "area": [1000, 1500, 2000, 2500, 3000, 3500, 4000],
    "bedrooms": [2, 3, 3, 4, 4, 5, 5],
    "price": [20, 30, 40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)
X = df[["area", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)
st.subheader("Enter House Details")

area = st.slider("Area (sq ft)", 500, 5000, 1000)
bedrooms = st.slider("Bedrooms", 1, 10, 2)
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms]])
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Price: {prediction[0]:.2f} Lakhs")
if st.checkbox("Show Training Data"):
    st.write(df)

st.markdown("---")
st.caption("Made with ❤️ using Machine Learning & Streamlit")
