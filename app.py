import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load Linear Regression Model
@st.cache_resource
def load_model():
    with open("Liner_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading Liner_model.pkl: {e}")
    st.stop()

# Header
st.title("🏠 Real Estate Price Valuation Engine")
st.write("Fill in the property parameters below to estimate house market values.")
st.markdown("---")

# Organized inputs in 3 distinct columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 General & Layout")
    prop_id = st.number_input("Property ID", value=1, step=1)
    date_val = st.number_input("Date Code (e.g. YYYYMMDD)", value=20141013, step=1)
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3, step=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, value=2.0, step=0.25)
    floors = st.number_input("Number of Floors", min_value=1.0, value=1.0, step=0.5)
    waterfront = st.selectbox("Waterfront Present", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    views = st.number_input("Number of Views", min_value=0, max_value=4, value=0, step=1)

with col2:
    st.subheader("📐 Area & Dimensions")
    living_area = st.number_input("Living Area (sqft)", min_value=0.0, value=2000.0, step=50.0)
    lot_area = st.number_input("Lot Area (sqft)", min_value=0.0, value=5000.0, step=100.0)
    area_above = st.number_input("Area excluding Basement (sqft)", min_value=0.0, value=1500.0, step=50.0)
    area_basement = st.number_input("Area of Basement (sqft)", min_value=0.0, value=500.0, step=50.0)
    living_area_renov = st.number_input("Living Area Renovated (sqft)", min_value=0.0, value=2000.0, step=50.0)
    lot_area_renov = st.number_input("Lot Area Renovated (sqft)", min_value=0.0, value=5000.0, step=100.0)
    condition = st.slider("Condition Rating", min_value=1, max_value=5, value=3)

with col3:
    st.subheader("📍 Location & History")
    grade = st.slider("Grade Rating", min_value=1, max_value=13, value=7)
    yr_built = st.number_input("Built Year", min_value=1800, max_value=2026, value=1995, step=1)
    yr_renovated = st.number_input("Renovation Year (0 if none)", min_value=0, max_value=2026, value=0, step=1)
    zipcode = st.number_input("Postal Code", value=98101, step=1)
    lat = st.number_input("Latitude", value=47.5112, format="%.4f")
    long_val = st.number_input("Longitude", value=-122.257, format="%.4f")
    schools = st.number_input("Schools Nearby", min_value=0, value=2, step=1)
    airport_dist = st.number_input("Distance from Airport (km)", min_value=0.0, value=15.0, step=0.5)

st.markdown("---")

# Prediction Execution
if st.button("Calculate Property Value", type="primary", use_container_width=True):
    # Vector ordered exactly as required by the model
    features = np.array([[
        prop_id, date_val, bedrooms, bathrooms, living_area, lot_area,
        floors, waterfront, views, condition, grade, area_above,
        area_basement, yr_built, yr_renovated, zipcode, lat, long_val,
        living_area_renov, lot_area_renov, schools, airport_dist
    ]])
    
    try:
        prediction = model.predict(features)[0]
        st.success(f"### Estimated Market Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
