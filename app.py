import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Use Streamlit caching only when in interactive mode.
if os.getenv("STREAMLIT_RUN", "False") == "True":
    cache_decorator = st.cache_data
else:
    # Dummy decorator for non-Streamlit environments.
    def cache_decorator(func):
         return func

@cache_decorator
def load_data():
    # Read the CSV file; adjust encoding if necessary.
    df = pd.read_csv("AmesHousing.csv", encoding="latin1")
    # Rename columns from the CSV to the expected names.
    rename_dict = {
        "Overall Qual": "OverallQual",
        "Gr Liv Area": "GrLivArea",
        "Year Built": "YearBuilt"
    }
    df.rename(columns=rename_dict, inplace=True)
    return df

@cache_decorator
def train_model(df):
    # Define expected features.
    features = ['OverallQual', 'GrLivArea', 'YearBuilt']
    # Check for required columns.
    missing_cols = [col for col in features + ['SalePrice'] if col not in df.columns]
    if missing_cols:
        msg = f"Missing columns in data: {missing_cols}"
        if os.getenv("STREAMLIT_RUN", "False") == "True":
            st.error(msg)
        else:
            print(msg)
        return None
    # Drop rows with missing values.
    df = df.dropna(subset=features + ['SalePrice'])
    X = df[features]
    y = df['SalePrice']
    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train model.
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def streamlit_app():
    st.title("Ames Housing Price Prediction")
    st.write("This web app predicts housing prices based on selected features from the Ames Housing dataset.")
    
    df = load_data()
    model = train_model(df)
    if model is None:
        st.error("Model could not be trained due to missing data.")
        return
    
    st.sidebar.header("Input Features")
    overall_qual = st.sidebar.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=5)
    gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=300, value=1500)
    year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=1970)
    
    # Create a DataFrame for prediction with the correct feature names.
    input_features = pd.DataFrame([[overall_qual, gr_liv_area, year_built]],
                                  columns=['OverallQual', 'GrLivArea', 'YearBuilt'])
    
    if st.sidebar.button("Predict Price"):
        prediction = model.predict(input_features)
        st.write(f"### Predicted Sale Price: ${prediction[0]:,.2f}")
    
    st.write("### Dataset Overview")
    st.dataframe(df.head())

def fallback_mode():
    print("Running in fallback mode (non-Streamlit environment)")
    df = load_data()
    model = train_model(df)
    if model is None:
        print("Model could not be trained due to missing data.")
        return
    
    print("Dataset Preview:")
    print(df.head())
    
    # Default input values.
    overall_qual = 5
    gr_liv_area = 1500
    year_built = 1970
    input_features = pd.DataFrame([[overall_qual, gr_liv_area, year_built]],
                                  columns=['OverallQual', 'GrLivArea', 'YearBuilt'])
    
    prediction = model.predict(input_features)
    print(f"Predicted Sale Price for default input: ${prediction[0]:,.2f}")

if __name__ == '__main__':
    # Check environment to decide mode.
    if os.getenv("STREAMLIT_RUN", "False") == "True":
        streamlit_app()
    else:
        fallback_mode()
