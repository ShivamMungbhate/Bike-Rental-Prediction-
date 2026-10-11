# Bike Rental Prediction System

A machine learning project that predicts hourly bike rental demand from time, calendar, and weather conditions.

## Live app

Try the deployed Streamlit app here:

**[Open Bike Rental Prediction System](https://epd6jvv6vjsyqq2uqsjccv.streamlit.app/)**

## Project objective

The goal is to estimate the number of bikes that may be rented during a selected hour. The app lets users enter time and weather conditions, then displays an estimated rental count and a chart of predicted demand across all 24 hours.

## Features

- Predicts hourly bike rentals.
- Accepts calendar and weather inputs through a Streamlit interface.
- Selects the season automatically from the selected month.
- Displays a 24-hour demand forecast chart.
- Uses a Random Forest regression model.

## Technology stack

- **Python** — project programming language
- **Pandas** — reads and prepares the dataset
- **Scikit-learn** — builds the preprocessing and machine learning pipeline
- **One-Hot Encoding** — converts categorical features into model-ready values
- **Random Forest Regressor** — predicts hourly rental counts
- **Streamlit** — provides the interactive web app

## Dataset

This project uses the hourly file (`hour.csv`) from the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

The target column is `cnt`, the total number of rentals. The model uses time, calendar, and weather features. The columns `casual` and `registered` are excluded from the inputs because they add up to `cnt` and would reveal the target.

Weather measurements use the dataset’s normalized 0–1 scale.

## Model evaluation

In one chronological 80/20 evaluation run, the model was trained on 13,903 records and evaluated on 3,476 later records.

| Metric | Result | Meaning |
|---|---:|---|
| MAE | 55.01 rentals | Average absolute prediction error |
| RMSE | 82.90 rentals | Prediction error metric that penalizes larger errors more |
| R² | 0.859 | How much variation in rental counts the model explained in this evaluation |

These are results from that evaluation run, not a guarantee of future prediction accuracy.

## Run locally

1. Clone or download this repository.
2. Make sure `app.py`, `hour.csv`, and `requirements.txt` are in the project folder.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

