import streamlit as st
import pandas as pd
import joblib
import os
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Pollution Dashboard", layout="wide")

st.title("🌍 Environmental Monitoring Dashboard")

# -------------------------------
# LOAD DATA (ROBUST PATH)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "Pollution_Weather_dataset.csv")

df = pd.read_csv(data_path)

# -------------------------------
# LOAD MODELS
# -------------------------------
rf_model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
dt_model = joblib.load(os.path.join(BASE_DIR, "dt_model.pkl"))
xgb_model = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
cols = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

# -------------------------------
# LOCATION INPUT
# -------------------------------
st.subheader("📍 Select Location")

mode = st.radio("Choose Input Method", ["City/Country", "Coordinates"])

# -------------------------------
# CITY / COUNTRY MODE
# -------------------------------
if mode == "City/Country":
    countries = sorted(df['Country'].dropna().unique())
    country = st.selectbox("Select Country", countries)

    cities = sorted(df[df['Country'] == country]['City'].dropna().unique())
    city = st.selectbox("Select City", cities)

    filtered = df[(df['Country'] == country) & (df['City'] == city)].iloc[0]

# -------------------------------
# LAT / LONG MODE
# -------------------------------
else:
    lat = st.number_input("Latitude", value=20.0)
    lon = st.number_input("Longitude", value=77.0)

    df['dist'] = ((df['Latitude'] - lat)**2 + (df['Longitude'] - lon)**2)
    filtered = df.loc[df['dist'].idxmin()]

# -------------------------------
# SHOW DATA USED
# -------------------------------
st.subheader("📊 Environmental Data Used")
st.dataframe(filtered)

# -------------------------------
# PREPARE INPUT FOR MODEL
# -------------------------------
input_data = filtered.drop(
    ['pollution_source', 'City', 'Country', 'Timestamp', 'Source_API'],
    errors='ignore'
)

input_df = pd.DataFrame([input_data])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=cols, fill_value=0)

# -------------------------------
# MODEL SELECTION
# -------------------------------
st.subheader("🤖 Choose Model")

model_choice = st.selectbox(
    "Select Model",
    ["Random Forest", "Decision Tree", "XGBoost"]
)

# -------------------------------
# PREDICTION
# -------------------------------
if model_choice == "Random Forest":
    prediction = rf_model.predict(input_df)
    result = prediction[0]

elif model_choice == "Decision Tree":
    prediction = dt_model.predict(input_df)
    result = prediction[0]

else:
    prediction = xgb_model.predict(input_df)
    result = le.inverse_transform(prediction)[0]

# -------------------------------
# DISPLAY RESULT
# -------------------------------
st.subheader("🔍 Prediction Result")
st.success(f"Predicted Pollution Source: {result}")

# -------------------------------
# ACCURACY COMPARISON
# -------------------------------
st.subheader("📊 Model Accuracy Comparison")

results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [0.858, 0.858, 0.887]  # your results
})

st.bar_chart(results.set_index('Model'))

# -------------------------------
# MAP VISUALIZATION
# -------------------------------
st.subheader("🗺️ Pollution Map")

lat_val = filtered['Latitude']
lon_val = filtered['Longitude']

m = folium.Map(location=[lat_val, lon_val], zoom_start=6)

folium.Marker(
    [lat_val, lon_val],
    popup=f"{result}",
    tooltip="Pollution Source"
).add_to(m)

st_folium(m, width=700, height=500)

# -------------------------------
# OPTIONAL HEATMAP HTML
# -------------------------------
st.subheader("🔥 Heatmap")

map_path = os.path.join(BASE_DIR, "pollution_map.html")

if os.path.exists(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=500)
else:
    st.warning("Heatmap file not found")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Developed for Environmental Monitoring & Pollution Source Detection 🌱")