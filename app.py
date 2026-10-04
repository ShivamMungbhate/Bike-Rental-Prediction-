import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="Bike Rental Predictor", page_icon="🚲")

st.title("🚲 Bike Rental Predictor")
st.write("Choose the time and weather conditions to estimate hourly bike rentals.")


@st.cache_resource
def train_model():
    data = pd.read_csv("hour.csv")
    data = data.sort_values(["dteday", "hr"]).reset_index(drop=True)

    features = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"
    ]

    categorical_features = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit"
    ]
    numeric_features = ["temp", "atemp", "hum", "windspeed"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numbers", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    model.fit(data[features], data["cnt"])
    return model


try:
    model = train_model()
except FileNotFoundError:
    st.error("I can't find hour.csv. Put it in the same folder as app.py.")
    st.stop()

st.header("Enter conditions")

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox(
        "Season",
        options=[1, 2, 3, 4],
        format_func=lambda value: {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter",
        }[value],
    )
    year = st.selectbox("Year", options=[0, 1], format_func=lambda value: str(2011 + value))
    month = st.slider("Month", min_value=1, max_value=12, value=6)
    hour = st.slider("Hour of day", min_value=0, max_value=23, value=8)
    weekday = st.selectbox(
        "Weekday",
        options=list(range(7)),
        format_func=lambda value: [
            "Sunday", "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday"
        ][value],
    )

with col2:
    holiday = st.selectbox(
        "Holiday?",
        options=[0, 1],
        format_func=lambda value: "No" if value == 0 else "Yes",
    )
    workingday = st.selectbox(
        "Working day?",
        options=[0, 1],
        format_func=lambda value: "No" if value == 0 else "Yes",
    )
    weather = st.selectbox(
        "Weather",
        options=[1, 2, 3, 4],
        format_func=lambda value: {
            1: "Clear or partly cloudy",
            2: "Mist or cloudy",
            3: "Light snow or rain",
            4: "Heavy rain or snow",
        }[value],
    )
    temperature = st.slider("Temperature (0–1)", 0.0, 1.0, 0.60)
    feels_like = st.slider("Feels-like temperature (0–1)", 0.0, 1.0, 0.58)
    humidity = st.slider("Humidity (0–1)", 0.0, 1.0, 0.55)
    windspeed = st.slider("Wind speed (0–1)", 0.0, 1.0, 0.20)

if st.button("Predict rentals", type="primary"):
    new_conditions = pd.DataFrame([{
        "season": season,
        "yr": year,
        "mnth": month,
        "hr": hour,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weather,
        "temp": temperature,
        "atemp": feels_like,
        "hum": humidity,
        "windspeed": windspeed,
    }])

    prediction = max(0, round(model.predict(new_conditions)[0]))
    st.success(f"Expected bike rentals: **{prediction}**")