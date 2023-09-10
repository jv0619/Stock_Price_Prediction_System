# Importing Libraries

import streamlit as st

import pandas as pd
import numpy as np
import datetime
from pandas_datareader import data as pdr
import yfinance as yfin
import matplotlib.pyplot as plt
from plotly import graph_objs as go

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model

current_day = datetime.datetime.now()
prev_day = current_day - datetime.timedelta(days=1)

# Taking data from yesterday to 10 years back
end = prev_day.strftime("%Y-%m-%d")
start = (current_day - datetime.timedelta(days=365 * 10)).strftime("%Y-%m-%d")

st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker", "GOOG")
yfin.pdr_override()
df = pdr.get_data_yahoo(user_input, start=start, end=end)

# Describe the Data
st.subheader(f"Data from {start} to {end} ")
st.write(df)
st.write(df.describe())

# Resetting Index
df.reset_index(inplace=True)

# Visualizations
# Plotting graph for Open and Close Prices
st.subheader("Open and Close Pricing  vs Time Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.Date, y=df.Open, name="Stock Open", marker_color='green'))
fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name="Stock Close", marker_color='red'))
fig.layout.update(title_text="Time Series data with Rangeslider",
                  xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Plotting graph for Moving Averages
st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name="Stock Open"))
fig.add_trace(go.Scatter(x=df.Date, y=ma100, name="100 Days Moving Average"))
fig.add_trace(go.Scatter(x=df.Date, y=ma200, name="200 Days Moving Average"))
fig.layout.update(title_text="Moving Average of 100 Days",
                  xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# splitting and Fitting Data for model
data_training = pd.DataFrame(df['Close'][: int(len(df) * 0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7): int(len(df))])
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Loading the model
model = load_model('keras_model.h5')

# Taking Data to test
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

# Testing Part
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scale = scaler.scale_
scale_factor = 1 / scale
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
y_predicted = np.array(y_predicted)
y_predicted.reshape(y_predicted.shape[1], y_predicted.shape[0])

# Plotting the graph for Predictions and Real

x_values = df['Date'][int(len(df)*0.7) - 100: ]
st.subheader("Actual Vs Predicted Stock")
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values, y=y_test, name="Actual Price"))
fig.add_trace(go.Scatter(x=x_values, y=y_predicted.flatten(), name="Predicted Price", marker_color='red'))
fig.layout.update(title_text="Time Series data with Rangeslider",
                  xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

