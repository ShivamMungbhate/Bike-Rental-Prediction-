import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv("hour.csv")

data = data.sort_values(["dteday", "hr"]).reset_index(drop=True)

# Select inputs for the prediction
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
X = data[features]
y = data["cnt"]

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

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

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"Dataset rows: {len(data)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print("\nModel evaluation:")
print(f"MAE:  {mae:.2f} bikes")
print(f"RMSE: {rmse:.2f} bikes")
print(f"R²:   {r2:.3f}")

results = pd.DataFrame(
    {
        "Actual rentals": y_test.iloc[:10].values,
        "Predicted rentals": np.round(predictions[:10]).astype(int),
    }
)

print("\nFirst 10 test predictions:")
print(results.to_string(index=False))

print("\nEnter conditions for a bike rental prediction:")

season = int(input("Season (1=spring, 2=summer, 3=fall, 4=winter): "))
year = int(input("Year (0=2011, 1=2012): "))
month = int(input("Month (1-12): "))
hour = int(input("Hour (0-23): "))
holiday = int(input("Holiday? (0=no, 1=yes): "))
weekday = int(input("Weekday (0=Sunday, 6=Saturday): "))
workingday = int(input("Working day? (0=no, 1=yes): "))
weather = int(input("Weather situation (1-4): "))

temperature = float(input("Temperature (0-1): "))
feels_like = float(input("Feels-like temperature (0-1): "))
humidity = float(input("Humidity (0-1): "))
windspeed = float(input("Wind speed (0-1): "))

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

expected_rentals = model.predict(new_conditions)[0]
print(f"\nExpected bike rentals: {round(expected_rentals)}")