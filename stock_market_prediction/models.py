import pandas as pd  # Import pandas for data manipulation
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA model from statsmodels for time series forecasting
import itertools  # Import itertools for creating combinations of parameters
import streamlit as st  # Import Streamlit for web app development

@st.cache_data  # Decorator to cache the results of this function in Streamlit, improving load times for repetitive calls
def fit_arima_model(df_train, df_test, column):
    """
    Fits an ARIMA model to the training data and finds the best parameter combination based on the Akaike Information Criterion (AIC).

    Parameters:
    - df_train (pd.DataFrame): The training dataset.
    - df_test (pd.DataFrame): The testing dataset, unused in this function.
    - column (str): The name of the column in the DataFrame to fit the ARIMA model on.

    Returns:
    - best_model: The ARIMA model fitted with the best parameter combination.
    - best_comb: A tuple of the best parameter combination (p, d, q) found.
    """
    # Defining ranges for ARIMA model parameters
    p = q = range(1, 6)  # Range of values for p and q parameters
    d = range(0, 5)  # Range of values for d parameter

    # Generating all combinations of p, d, and q
    pdq = list(itertools.product(p, d, q))

    best_comb = pdq[0]  # Initialize best_comb with the first combination
    best_score = 100000  # Initialize best_score with a very high value
    best_model = None  # Initialize best_model as None

    # Iterate over all parameter combinations
    for param in pdq:
        try:
            # Try fitting ARIMA model with the current parameter combination
            arima = ARIMA(df_train[column], order=param, enforce_stationarity=True, enforce_invertibility=True)
            results = arima.fit()

            # Update the best_score, best_comb, and best_model if the current model's AIC is lower
            if results.aic < best_score:
                best_score = results.aic
                best_comb = param
                best_model = results

        except Exception as e:
            # Print error message if model fitting fails
            print(f'ARIMA {param} – AIC: {e}')
            continue  # Skip to the next iteration

    return best_model, best_comb


def calc_smoothed_average(input_data: pd.DataFrame, column: str, ma_window: int = 200) -> pd.DataFrame:
    """
    Calculates a smoothed average (simple moving average) for a specified column in the input DataFrame.

    Parameters:
    - input_data (pd.DataFrame): The DataFrame containing the data.
    - column (str): The name of the column to calculate the smoothed average for.
    - ma_window (int): The window size for the moving average calculation.

    Returns:
    - pd.DataFrame: A DataFrame containing the smoothed averages.
    """
    df = input_data.copy()  # Make a copy of the input DataFrame
    # Calculate the moving average and store in a new column
    df['sma_forecast'] = df[column].rolling(ma_window).mean()
    return df[['sma_forecast']]  # Return the DataFrame with the smoothed averages
