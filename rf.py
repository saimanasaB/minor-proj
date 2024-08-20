import streamlit as st
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# Convert 'Year' and 'Month' to datetime format and create a 'Date' column
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

# Streamlit app title
st.title('CPI Forecasting with ARIMA and SARIMA')

# Sidebar for user input
st.sidebar.header('User Input Parameters')
sector = st.sidebar.selectbox('Select Sector', df['Sector'].unique())
category = st.sidebar.selectbox('Select Category', df.columns[4:])  # Assuming the first 4 columns are not categories

# Filter data based on user input
filtered_data = df[df['Sector'] == sector]

# Build time series data for ARIMA and SARIMA
time_series = filtered_data[['Date', category]].set_index('Date').asfreq('MS')[category]

# Drop missing values
time_series = time_series.dropna()

# Sidebar for forecasting
n_periods = st.sidebar.slider('Select number of months to forecast (1-36)', 1, 36)
forecast_button = st.sidebar.button('Predict')

# Create placeholders for the plots
plot_arima_placeholder = st.empty()
plot_sarima_placeholder = st.empty()
metrics_placeholder = st.empty()

if forecast_button:
    # ARIMA Model
    model_arima = pm.auto_arima(time_series, seasonal=False, stepwise=True)
    forecast_arima, conf_int_arima = model_arima.predict(n_periods=n_periods, return_conf_int=True)
    
    # Create a date range for the forecasted values
    future_dates_arima = pd.date_range(start=time_series.index[-1], periods=n_periods+1, freq='MS')[1:]

    # ARIMA Plot
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=time_series.index, y=time_series, mode='lines', name='Observed', line=dict(color='blue')))
    fig_arima.add_trace(go.Scatter(x=future_dates_arima, y=forecast_arima, mode='lines', name='Forecast', line=dict(color='orange')))
    fig_arima.add_trace(go.Scatter(
        x=list(future_dates_arima) + list(reversed(future_dates_arima)),
        y=list(conf_int_arima[:, 0]) + list(reversed(conf_int_arima[:, 1])),
        fill='toself',
        fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    fig_arima.update_layout(title=f'ARIMA Forecast for {category} in {sector} Sector', xaxis_title='Date', yaxis_title=category)
    plot_arima_placeholder.plotly_chart(fig_arima)

    # SARIMA Model
    model_sarima = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit_sarima = model_sarima.fit(disp=False)
    forecast_sarima = model_fit_sarima.predict(start=len(time_series), end=len(time_series) + n_periods - 1, dynamic=False)
    
    # Create a date range for the forecasted values
    future_dates_sarima = pd.date_range(start=time_series.index[-1], periods=n_periods+1, freq='MS')[1:]

    # SARIMA Plot
    fig_sarima = go.Figure()
    fig_sarima.add_trace(go.Scatter(x=time_series.index, y=time_series, mode='lines', name='Observed', line=dict(color='blue')))
    fig_sarima.add_trace(go.Scatter(x=future_dates_sarima, y=forecast_sarima, mode='lines', name='Forecast', line=dict(color='orange')))
    fig_sarima.update_layout(title=f'SARIMA Forecast for {category} in {sector} Sector', xaxis_title='Date', yaxis_title=category)
    plot_sarima_placeholder.plotly_chart(fig_sarima)

    # Align forecast with test data for metrics calculation
    forecast_arima = model_arima.predict(n_periods=n_periods)
    forecast_sarima = model_fit_sarima.predict(start=len(time_series), end=len(time_series) + n_periods - 1)

   