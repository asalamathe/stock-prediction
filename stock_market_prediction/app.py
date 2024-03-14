import streamlit as st  # Import Streamlit for web app development
import pandas as pd  # Import pandas for data manipulation
import plotly.express as px  # Import Plotly Express for easy plotting (unused in this script)
import plotly.graph_objects as go  # Import Plotly Graph Objects for customized plotting
from .models import calc_smoothed_average, fit_arima_model  # Import custom functions for stock market prediction
import warnings  # Import warnings to control warning messages
from sklearn.metrics import r2_score, mean_absolute_error  # Import metrics for model evaluation
import pandas as pd  # Import pandas for data manipulation
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA model from statsmodels for time series forecasting
import itertools  # Import itertools for creating combinations of parameters
import streamlit as st  # Import Streamlit for web app development

warnings.filterwarnings("ignore")  # Ignore warning messages for a cleaner presentation

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


@st.cache_data  # Cache data to improve load times in Streamlit applications
def get_data(train_size, test_size_weeks):
    """
    Load and split the dataset into training and testing sets based on specified sizes.

    Parameters:
    - train_size (int): The size of the training dataset in years.
    - test_size (int): The size of the testing dataset in weeks.

    Returns:
    - df_train (DataFrame): The training dataset.
    - df_test (DataFrame): The testing dataset.
    - df (DataFrame): The complete dataset.
    """
    df = pd.read_parquet('data/data.parquet')  # Load dataset from a parquet file
    df.index = pd.DatetimeIndex(df.index)  # Ensure the index is a DateTimeIndex for time series analysis

    # Split data into training and testing sets based on the specified sizes
    df_train = df.copy()
    df_train = df_train[pd.Timestamp('2004-03-11'):pd.Timestamp('2004-03-11') + pd.Timedelta(days=train_size*365)]

    df_test = df.copy()
    df_test = df_test[pd.Timestamp('2004-03-11') + pd.Timedelta(days=train_size*365):pd.Timestamp('2004-03-11') + pd.Timedelta(days=(train_size*365+test_size_weeks*7))]

    return df_train, df_test, df

def main():
    """
    The main function where Streamlit UI components are defined and stock market predictions are visualized.
    """
    st.title('Aktienmarkt Vorhersage')  # Set the title of the Streamlit application

    st.markdown("---")  # Add a markdown separator for layout

    with st.expander('Einstellungen', expanded=True):  # Create an expandable settings menu
        empty = st.empty()  # Placeholder for dynamic content
        cols = st.columns(2)  # Split settings into two columns

        with cols[0]:  # Column for training data settings
            train_length = st.slider('Trainingsdaten (in Jahren)', 1, 19, 18)  # Slider for training data length

        with cols[1]:  # Column for testing data settings
            # Adjust the range and default value of test data length based on training data length
            if train_length == 19:
                test_length = 1
                st.slider('Testdaten (in Jahren)', 0, 1, 1, disabled=True)  # Disabled slider if training data uses all but 1 year
            else:
                test_length_weeks = st.slider('Testdaten (in Wochen)', 1, 53, 2)

        with empty:  # Display information about the selected time period
            st.info(f'Zeitraum {train_length} Jahre')

        cols = st.columns(2)  # Another set of columns for additional settings

        with cols[0]:  # Column for selecting the stock type
            options = {"ruhig": "calm", "starke Außreißer": "outliers", "volatile": "volatile"}
            stock = st.selectbox('Aktie', list(options.keys()))  # Dropdown for stock selection
            selected_stock = options[stock]

        with cols[1]:  # Column for window size input
            window = st.number_input('Fenstergröße', 1, 1000, 200)  # Input for window size

        calculate = st.button('Berechnen', key='calculate', use_container_width=True)  # Button to trigger calculations

    if calculate:  # Actions to perform when the calculate button is pressed
        # Load and split data based on user selections
        df_train, df_test, df = get_data(train_length, test_length_weeks)

        # Calculate smoothed average and fit ARIMA model to the training data
        smoothed_average = calc_smoothed_average(df_train, "average_" + selected_stock, window)
        best_model, best_comb = fit_arima_model(df_train, df_test, "average_" + selected_stock)

        # Predict future values with the ARIMA model
        preds = best_model.get_forecast(steps=df_test.shape[0])
        preds_ci = preds.conf_int()

        # Calculate performance metrics
        r2 = r2_score(df_test["average_" + selected_stock], preds.predicted_mean)
        mae = mean_absolute_error(df_test["average_" + selected_stock], preds.predicted_mean)

        # Calculate residues for smoothed and ARIMA predictions
        residues_smoothed_test = df_test["average_" + selected_stock] - smoothed_average["sma_forecast"].values[-1]
        residues_arima_test = df_test["average_" + selected_stock].values - preds.predicted_mean
            
        # Visualization and performance metrics sections
            
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
                        <div style="text-align: center;">
                            <h1 style="font-size: 50px; margin-bottom: -10px; margin-top: 50px;">{r2:.4f}</h1>
                            <p style="color: gray; font-size: 20px;">Bestimmtheitsmaß</p>
                        </div>
                        """, unsafe_allow_html=True)
            
        with cols[1]:
            st.markdown(f"""
                        <div style="text-align: center;">
                            <h1 style="font-size: 50px; margin-bottom: -10px; margin-top: 50px;">{mae:.4f}</h1>
                            <p style="color: gray; font-size: 20px;">Gütemaß</p>
                        </div>
                        """, unsafe_allow_html=True)
            
        st.markdown("---")
            
        st.markdown(f"""<h2 style="margin-bottom: -30px;">Aktienverlauf</h2>""", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train.index, y=df_train["average_" + selected_stock], line=dict(color="#ffaa00"), name="Train Data"))
        fig.add_trace(go.Scatter(x=df_test.index, y=df_test["average_" + selected_stock], line=dict(color="#00ff00"), name="Test Data"))
        fig.update_layout(title="Stock Market Data", xaxis_title="Time", yaxis_title="Average Value", legend_title="Legend")
        st.plotly_chart(fig, use_container_width=True)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train.index, y=smoothed_average["sma_forecast"], line=dict(color="blue"), name="Smoothed Average"))
        fig.add_trace(go.Scatter(x=df_train.index, y=df_train["average_" + selected_stock], line=dict(color="gray", width=1), name="Actual Values"))
        fig.update_layout(title="Smoothed Average", xaxis_title="Time", yaxis_title="Average Value", legend_title="Legend")
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train.index, y=df_train["average_" + selected_stock], mode='lines', name='Training Data', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=df_train.index, y=best_model.get_prediction().predicted_mean, mode='lines', name='ARIMA Predictions', line=dict(color='coral')))
        fig.update_layout(title='Training Data and ARIMA Model Predictions', xaxis_title='Time', yaxis_title='Average Value', legend_title='Legend')
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_test.index, y=df_test["average_" + selected_stock], mode='lines', name='Testing Data', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=df_test.index, y=preds.predicted_mean, mode='lines', name='ARIMA Predictions', line=dict(color='coral')))
        fig.add_trace(go.Scatter(x=df_test.index, y=[smoothed_average["sma_forecast"].values[-1]] * df_test.shape[0], mode='lines', name='Smoothed Average', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_test.index, y=preds_ci.iloc[:, 0], mode='lines', name='ARIMA Lower Confidence Bound', line=dict(color='peachpuff', dash='dot')))
        fig.add_trace(go.Scatter(x=df_test.index, y=preds_ci.iloc[:, 1], mode='lines', fill='tonexty', name='Upper Confidence Bound', line=dict(color='peachpuff', dash='dot')))
        fig.update_layout(title='Forecasts', xaxis_title='Time', yaxis_title='ARIMA Average Value', legend_title='Legend')
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_test.index, y=residues_smoothed_test, mode='lines', name='Residues Smoothed Average', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_test.index, y=residues_arima_test, mode='lines', name='Residues ARIMA', line=dict(color='coral')))
        fig.update_layout(title='Residues', xaxis_title='Time', yaxis_title='Residues', legend_title='Legend')
        st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
