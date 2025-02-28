import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset from the local Excel file
df = pd.read_excel("AmesHousing.xlsx")

# Basic data preprocessing:
# For demonstration, drop rows with missing values.
df = df.dropna()

# Select features and target.
# Here, we use 'GrLivArea' (above ground living area) and 'OverallQual' (overall quality)
# to predict 'SalePrice'. Adjust these features based on your exploration of the dataset.
features = ['GrLivArea', 'OverallQual']
target = 'SalePrice'

# Ensure the selected columns exist in the dataset
if not set(features + [target]).issubset(df.columns):
    st.error("The required columns are not present in the dataset.")
    st.stop()

# Prepare the data
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit web app layout
st.title("Ames Housing Price Predictor")
st.write("Enter the details of the house to predict its sale price:")

# Input widgets for user to provide feature values
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500, step=50)
overall_qual = st.slider("Overall Quality (1 - 10)", min_value=1, max_value=10, value=5)

# Predict when button is clicked
if st.button("Predict Sale Price"):
    # Create a DataFrame for the input features
    input_features = np.array([[gr_liv_area, overall_qual]])
    prediction = model.predict(input_features)[0]
    st.success(f"Predicted Sale Price: ${prediction:,.2f}")

# Optional: Display model performance (if desired)
score = model.score(X_test, y_test)
st.write(f"Model R² score on test data: {score:.2f}")
