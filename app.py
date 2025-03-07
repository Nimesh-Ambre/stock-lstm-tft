import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Set up Streamlit app
st.set_page_config(page_title="Stock Market Prediction with LSTM & TFT", layout="wide")
st.title("📈 Stock Market Prediction using LSTM & TFT")
st.markdown("### Analyze stock trends and predict future prices with AI")

# Sidebar - User Selection
stock_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
time_period = st.sidebar.selectbox("Select Time Range", ["1y", "5y", "10y", "max"])

# Fetch stock data
@st.cache_data
def get_stock_data(ticker, period):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            raise ValueError(f"No historical data found for {ticker}")
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

stock_data = get_stock_data(stock_symbol, time_period)

if stock_data.empty:
    st.warning("⚠️ No data available for the selected stock.")
    st.stop()

st.subheader(f"📊 {stock_symbol} Stock Performance")
st.line_chart(stock_data['Close'])

# ---------------------------------- TFT PREPROCESSING ----------------------------------

# Function to apply TFT transformation
def preprocessing_fn(inputs):
    outputs = {}
    outputs['Close_normalized'] = tft.scale_to_z_score(inputs['Close'])
    return outputs

# Apply TFT preprocessing
def apply_tft_transformation(stock_data):
    raw_data = {'Close': stock_data['Close'].values.astype(np.float32)}
    transformed_data = preprocessing_fn(raw_data)
    return transformed_data

tft_data = apply_tft_transformation(stock_data)

# ----------------------------- LSTM MODEL TRAINING ------------------------------------

# Prepare data for LSTM
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

X, y, scaler = prepare_data(stock_data)

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model
model = build_lstm_model((X.shape[1], 1))
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Predict future stock price
def predict_price(model, scaler, stock_data, look_back=60):
    inputs = stock_data['Close'].tail(look_back).values.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    inputs = np.reshape(inputs, (1, look_back, 1))
    predicted_price = model.predict(inputs)
    return scaler.inverse_transform(predicted_price)[0][0]

predicted_price = predict_price(model, scaler, stock_data)

st.subheader(f"🔮 Predicted Closing Price for Next Day: ${predicted_price:.2f}")
