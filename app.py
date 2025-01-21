import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

@st.cache_resource
def load_model():
    return joblib.load('trained_rf_model.pkl')

def predict_insurance_charges(model, input_data):
    prediction = model.predict([input_data])
    return np.exp(prediction[0])

st.title("Insurance Charges Prediction Dashboard")
st.markdown("## Explore How Your Choices Affect Insurance Charges")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=30, step=1, help="Select the individual's age.")
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, help="Enter the individual's Body Mass Index.")
with col2:
    children = st.slider("Number of Children", min_value=0, max_value=10, value=0, step=1, help="Select the number of children.")
    sex_male = st.selectbox("Gender", options=["Male", "Female"], index=1)

smoker_yes = st.selectbox("Smoker", options=["Yes", "No"], index=1)
region = st.selectbox("Region", options=["Northwest", "Northeast", "Southeast", "Southwest"], index=0)

sex_male = 1 if sex_male == "Male" else 0
smoker_yes = 1 if smoker_yes == "Yes" else 0
region_encoding = {"Northwest": [1, 0, 0], "Southeast": [0, 1, 0], "Southwest": [0, 0, 1], "Northeast": [0, 0, 0]}
region_values = region_encoding[region]
input_data = [age, bmi, children, sex_male, smoker_yes] + region_values

model = load_model()

if st.button("Predict"):
    try:
        prediction = predict_insurance_charges(model, input_data)
        st.subheader(f"Predicted Insurance Charges: ${prediction:,.2f}")

        

        # Download button
        data = {
            "Age": [age],
            "BMI": [bmi],
            "Children": [children],
            "Gender (Male)": [sex_male],
            "Smoker (Yes)": [smoker_yes],
            "Region": [region],
            "Predicted Charges": [prediction]
        }
        df = pd.DataFrame(data)
        st.download_button(
            label="Download Results as CSV",
            data=df.to_csv(index=False),
            file_name="insurance_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
# Extract feature importance from the Random Forest model
import pandas as pd

# Assuming your model is a Random Forest
feature_importance_values = model.feature_importances_
feature_names = ["Age", "BMI", "Number of Children", "Gender", "Smoker", "Region_Northwest", "Region_Southeast", "Region_Southwest"]  # Replace with actual feature names used in your model

# Create a dictionary of feature names and their importance
feature_importances = pd.Series(feature_importance_values, index=feature_names).to_dict()

import plotly.express as px

# Convert feature importances into a sorted Pandas DataFrame
import pandas as pd
import plotly.express as px

# Create a DataFrame and sort by importance
feature_importances_df = pd.DataFrame({
    "Feature": list(feature_importances.keys()),
    "Importance": list(feature_importances.values())
}).sort_values(by="Importance", ascending=False)

# Create a bar chart for sorted feature importance
fig = px.bar(
    feature_importances_df,
    x="Feature",
    y="Importance",
    title="Feature Importance for Insurance Predictions",
    labels={"Feature": "Features", "Importance": "Importance"},
    text_auto=True,
)

# Display the chart in Streamlit
st.plotly_chart(fig)

