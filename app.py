import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="Bike Rental Predictor",
    page_icon="🚲",
    layout="wide",
)


SEASON_BY_MONTH = {
    1: 1, 2: 1, 12: 1,
    3: 2, 4: 2, 5: 2,   
    6: 3, 7: 3, 8: 3,  
    9: 4, 10: 4, 11: 4, 
}

SEASON_NAMES = {
    1: "Winter",
    2: "Spring",
    3: "Summer",
    4: "Fall",
}

WEEKDAY_NAMES = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(ellipse at top left, #13232c 0%, transparent 42%),
                #080b10;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2rem 2.2rem;
            border: 1px solid #263a45;
            border-radius: 20px;
            color: #f8fafc;
            background: linear-gradient(110deg, #10212a, #123b3a);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin: 0;
            color: #f8fafc;
            font-size: 2.2rem;
        }

        .hero p {
            margin: 0.65rem 0 0;
            color: #b8c9d2;
        }

        .section-title {
            color: #edf2f7;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        div[data-testid="stForm"],
        div[data-testid="stMetric"],
        .result-card {
            color: #edf2f7;
            background: #121923;
            border: 1px solid #273444;
            border-radius: 16px;
        }

        div[data-testid="stForm"] {
            padding: 1.2rem;
        }

        div[data-testid="stMetric"] {
            padding: 1rem;
        }

        .result-card {
            padding: 1.4rem;
        }

        .result-card h3 {
            color: #edf2f7;
        }

        .result-card p {
            color: #b8c4d1;
        }

        button[kind="primary"] {
            min-height: 3rem;
            border-radius: 10px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def train_model():
    data = pd.read_csv("hour.csv")
    data = data.sort_values(["dteday", "hr"]).reset_index(drop=True)

    features = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
    ]

    categorical_features = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]

    numeric_features = [
        "temp",
        "atemp",
        "hum",
        "windspeed",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numbers",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(data[features], data["cnt"])
    historical_peak = int(data["cnt"].max())

    return model, historical_peak



st.markdown(
    """
    <div class="hero">
        <h1>🚲 Bike Rental Predictor</h1>
        <p>
            Plan for the ride ahead. Estimate hourly bike demand from time and weather.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    model, historical_peak = train_model()
except FileNotFoundError:
    st.error("Couldn't find `hour.csv`. Put it in the same folder as `app.py`.")
    st.stop()
except Exception as error:
    st.error(f"Couldn't load the data or train the model: {error}")
    st.stop()


left_column, right_column = st.columns([1.5, 1], gap="large")


with left_column:
    st.markdown(
        '<div class="section-title">Set the conditions</div>',
        unsafe_allow_html=True,
    )
    st.caption("Choose the time and weather conditions for your estimate.")

    with st.form("prediction_form"):
        time_column, weather_column = st.columns(2)

        with time_column:
            st.markdown("#### 📅 Time")

            year = st.selectbox(
                "Year",
                [0, 1],
                format_func=lambda value: str(2011 + value),
            )

            month = st.selectbox(
                "Month",
                list(range(1, 13)),
                format_func=lambda value: pd.Timestamp(
                    2000, value, 1
                ).strftime("%B"),
            )

            
            season = SEASON_BY_MONTH[month]
            st.caption(f"Season selected automatically: {SEASON_NAMES[season]}")

            hour = st.slider(
                "Hour of day",
                min_value=0,
                max_value=23,
                value=8,
                help="0 is midnight; 12 is noon.",
            )

            weekday = st.selectbox(
                "Day of week",
                list(range(7)),
                format_func=lambda value: WEEKDAY_NAMES[value],
            )

        with weather_column:
            st.markdown("#### 🌦️ Weather")

            weather = st.selectbox(
                "Conditions",
                [1, 2, 3, 4],
                format_func=lambda value: {
                    1: "Clear or partly cloudy",
                    2: "Mist or cloudy",
                    3: "Light rain or snow",
                    4: "Heavy rain or snow",
                }[value],
            )

            temperature = st.slider(
                "Temperature (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.01,
            )

            feels_like = st.slider(
                "Feels-like temperature (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=0.58,
                step=0.01,
            )

            humidity = st.slider(
                "Humidity (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
            )

            windspeed = st.slider(
                "Wind speed (0–1)",
                min_value=0.0,
                max_value=1.0,
                value=0.20,
                step=0.01,
            )

        st.markdown("#### 🗓️ Day type")
        holiday_column, work_column = st.columns(2)

        with holiday_column:
            holiday = st.selectbox(
                "Public holiday?",
                [0, 1],
                format_func=lambda value: "No" if value == 0 else "Yes",
            )

        with work_column:
            workingday = st.selectbox(
                "Working day?",
                [0, 1],
                format_func=lambda value: "No" if value == 0 else "Yes",
            )

        submitted = st.form_submit_button(
            "✨  Estimate bike rentals",
            type="primary",
            use_container_width=True,
        )



if submitted:
    base_conditions = {
        "season": season,
        "yr": year,
        "mnth": month,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weather,
        "temp": temperature,
        "atemp": feels_like,
        "hum": humidity,
        "windspeed": windspeed,
    }

    hourly_conditions = pd.DataFrame(
        [
            {**base_conditions, "hr": forecast_hour}
            for forecast_hour in range(24)
        ]
    )

    hourly_predictions = model.predict(hourly_conditions)

    
    prediction = max(0, round(float(hourly_predictions[hour])))

    
    hourly_chart = pd.DataFrame(
        {
            "Hour": [
                f"{forecast_hour:02d}:00"
                for forecast_hour in range(24)
            ],
            "Predicted rentals": [
                max(0, round(float(value)))
                for value in hourly_predictions
            ],
        }
    )


with right_column:
    st.markdown(
        '<div class="section-title">Your estimate</div>',
        unsafe_allow_html=True,
    )
    st.caption("The estimate is for one hour.")

    if submitted:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.metric("Expected rentals", f"{prediction:,} bikes")

        if historical_peak > 0:
            st.progress(min(prediction / historical_peak, 1.0))

        st.caption(
            "The bar compares this estimate with the highest hourly count in the dataset."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.info(
            f"Estimate for {hour:02d}:00 on {WEEKDAY_NAMES[weekday]}."
        )
    else:
        st.markdown(
            """
            <div class="result-card">
                <h3>Ready when you are</h3>
                <p>
                    Choose the time and weather conditions, then select
                    <b>Estimate bike rentals</b> to see your prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )



if submitted:
    st.divider()
    st.subheader("Demand across the day")
    st.caption(
        "Predicted hourly rentals for the selected month, day type, and weather."
    )

    st.line_chart(
        hourly_chart,
        x="Hour",
        y="Predicted rentals",
        color="#2dd4bf",
        height=320,
    )

st.divider()
st.caption(
    "Model: Random Forest regression · Weather measurements use the dataset's normalized 0–1 scale."
)