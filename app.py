import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="ML House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction System")
st.write("This is a Machine Learning model that predicts house prices.")

# Sidebar
st.sidebar.header("About Project")
st.sidebar.write("""
- Model: Linear Regression  
- Type: Supervised Learning  
- Features: Area & Bedrooms  
""")

# Dataset
data = {
    "area": [800, 1000, 1500, 2000, 2500, 3000, 3500],
    "bedrooms": [1, 2, 2, 3, 4, 4, 5],
    "price": [15, 20, 30, 40, 50, 60, 75]
}

df = pd.DataFrame(data)

# Model
X = df[["area", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# Inputs
st.subheader("Enter Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", 500, 5000, 1000)

with col2:
    bedrooms = st.number_input("Bedrooms", 1, 10, 2)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms]])
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Price: {prediction[0]:.2f} Lakhs")

# Graph
st.subheader("📊 Data Visualization")
st.line_chart(df.set_index("area"))

# Dataset
if st.checkbox("Show Dataset"):
    st.write(df)
