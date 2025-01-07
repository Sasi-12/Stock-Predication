# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Title
st.title("Stock Price Prediction for Next Day")

# Instruction
st.write("Provide the closing prices for the past 10 days to predict the stock price for the next day.")

# Input for past 10 days of stock prices
past_prices = st.text_input("Enter the last 10 days' closing prices, separated by commas (e.g., 100.1, 102.3, 101.4, ...):")

if past_prices:
    try:
        # Parse input into a numpy array
        past_prices = np.array([float(price) for price in past_prices.split(",")])
        
        # Check if the input has exactly 10 values
        if len(past_prices) == 10:
            # Reshape and scale the data
            past_prices = past_prices.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            past_prices_scaled = scaler.fit_transform(past_prices)
            past_prices_scaled = past_prices_scaled.reshape(1, 10, 1)

            # Define the model (ensure it matches the one trained)
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(past_prices_scaled.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))

            # Load pre-trained weights (if saved) or train the model here
            model.load_weights("my_model.keras") 
            # Predict the next day price
            next_day_price_scaled = model.predict(past_prices_scaled)
            next_day_price = scaler.inverse_transform(next_day_price_scaled)

            # Display the prediction
            st.write(f"Predicted stock price for the next day: {next_day_price[0][0]:.2f}")
        else:
            st.write("Please enter exactly 10 closing prices.")

    except ValueError:
        st.write("Invalid input. Please ensure you enter 10 numerical values separated by commas.")

else:
    st.write("Enter the past 10 days' closing prices to get the prediction.")
