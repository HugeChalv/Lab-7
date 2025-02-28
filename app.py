import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset from the CSV file. Ensure that "AmesHousing.csv" is in the same directory.
@st.cache
def load_data():
    df = pd.read_csv("AmesHousing.csv")
    return df

# Train a regression model using selected features.
@st.cache
def train_model(df):
    # For simplicity, we use a few features: OverallQual, GrLivArea, and YearBuilt.
    features = ['OverallQual', 'GrLivArea', 'YearBuilt']
    # Drop rows with missing values in the selected columns.
    df = df.dropna(subset=features + ['SalePrice'])
    X = df[features]
    y = df['SalePrice']
    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the Linear Regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Ames Housing Price Prediction")
    st.write("This web app predicts housing prices based on selected features from the Ames Housing dataset.")

    # Load the dataset and train the model.
    df = load_data()
    model = train_model(df)
    
    # Sidebar for user inputs.
    st.sidebar.header("Input Features")
    overall_qual = st.sidebar.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=5)
    gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=300, value=1500)
    year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=1970)
    
    # Prepare the input for prediction.
    input_features = np.array([[overall_qual, gr_liv_area, year_built]])
    
    # When the predict button is clicked, display the prediction.
    if st.sidebar.button("Predict Price"):
        prediction = model.predict(input_features)
        st.write(f"### Predicted Sale Price: ${prediction[0]:,.2f}")
    
    # Display a short preview of the dataset.
    st.write("### Dataset Overview")
    st.dataframe(df.head())

if __name__ == '__main__':
    main()
