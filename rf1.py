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

# Define relevant categories for forecasting
relevant_categories = [
    'Cereals and products', 'Meat and fish', 'Egg', 'Milk and products',
    'Oils and fats', 'Fruits', 'Vegetables', 'Pulses and products',
    'Sugar and Confectionery', 'Spices', 'Non-alcoholic beverages',
    'Prepared meals, snacks, sweets etc.', 'Food and beverages',
    'Pan, tobacco and intoxicants', 'Clothing', 'Footwear',
    'Clothing and footwear', 'Fuel and light',
    'Household goods and services', 'Health',
    'Transport and communication', 'Recreation and amusement',
    'Education', 'Personal care and effects', 'Miscellaneous', 'General index'
]

# Streamlit app title
st.title('VALIDATION')

# Sidebar for user input
st.sidebar.header('User Input Parameters')
sector = st.sidebar.selectbox('Select Sector', df['Sector'].unique())
category = st.sidebar.selectbox('Select Category', relevant_categories)

# Filter data based on user input
filtered_data = df[df['Sector'] == sector]

# Build time series data for ARIMA and SARIMA
time_series = filtered_data[['Date', category]].set_index('Date').asfreq('MS')[category]

# Drop missing values
time_series = time_series.dropna()

# Split data into training and validation sets
train_data = time_series[time_series.index.year < 2023]
validation_data = time_series[time_series.index.year >= 2023]

# Sidebar for forecasting
n_periods = st.sidebar.slider('Select number of months to forecast (1-36)', 1, 36)
forecast_button = st.sidebar.button('Predict')

# Create placeholders for the plots
plot_arima_placeholder = st.empty()
plot_sarima_placeholder = st.empty()
metrics_placeholder = st.empty()
comparison_placeholder = st.empty()
validation_plot_placeholder = st.empty()

if forecast_button:
    # ARIMA Model
    model_arima = pm.auto_arima(train_data, seasonal=False, stepwise=True)
    forecast_arima, conf_int_arima = model_arima.predict(n_periods=len(validation_data), return_conf_int=True)
    
    # SARIMA Model
    model_sarima = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit_sarima = model_sarima.fit(disp=False)
    forecast_sarima = model_fit_sarima.predict(start=len(train_data), end=len(train_data) + len(validation_data) - 1, dynamic=False)
    
    # Validation Plot for ARIMA
    validation_fig_arima = go.Figure()
    validation_fig_arima.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Train Data', line=dict(color='blue')))
    validation_fig_arima.add_trace(go.Scatter(x=validation_data.index, y=validation_data, mode='lines', name='Validation Data', line=dict(color='green')))
    validation_fig_arima.add_trace(go.Scatter(x=validation_data.index, y=forecast_arima, mode='lines', name='ARIMA Forecast', line=dict(color='orange')))
    validation_fig_arima.update_layout(title=f'ARIMA Validation Forecast for {category} in {sector} Sector', xaxis_title='Date', yaxis_title=category)
    plot_arima_placeholder.plotly_chart(validation_fig_arima)

    # Validation Plot for SARIMA
    validation_fig_sarima = go.Figure()
    validation_fig_sarima.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Train Data', line=dict(color='blue')))
    validation_fig_sarima.add_trace(go.Scatter(x=validation_data.index, y=validation_data, mode='lines', name='Validation Data', line=dict(color='green')))
    validation_fig_sarima.add_trace(go.Scatter(x=validation_data.index, y=forecast_sarima, mode='lines', name='SARIMA Forecast', line=dict(color='orange')))
    validation_fig_sarima.update_layout(title=f'SARIMA Validation Forecast for {category} in {sector} Sector', xaxis_title='Date', yaxis_title=category)
    plot_sarima_placeholder.plotly_chart(validation_fig_sarima)

    
