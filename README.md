# Time Series Prediction using Stacked LSTM

This project aims to predict the future stock prices of AAPL (Apple Inc.) using Stacked LSTM (Long Short-Term Memory) neural networks. We'll be forecasting the closing stock prices based on historical data.

## Dataset
The dataset used for this project is the AAPL dataset, containing various features related to Apple Inc. stocks. For simplicity, we focused on the 'Close' feature. The dataset is split into training and testing sets in an 80:20 ratio. 

To prepare the dataset for forecasting:
1. We created a new dataset where each data point consists of the closing stock prices of the previous 100 days (features) and the closing stock price of the next day (output).
2. We skipped one day and repeated the process to create sequential data points.

## Model Architecture
We built a Stacked LSTM model with the following layers:
- LSTM layer with 50 units (returning sequences) and input shape (2, 1)
- Another LSTM layer with 50 units (returning sequences)
- Third LSTM layer with 50 units
- Dense layer with 1 unit (output)
The model is compiled with Mean Squared Error loss and Adam optimizer.

## Data Preprocessing
We used Min-Max Scaler to scale down the values of the dataset before feeding them into the model.

## Training
The model was trained for 100 epochs. After training, the Mean Squared Error (MSE) loss on the training set was 140, and on the testing set was 235.

## Evaluation
We plotted the original and predicted stock prices for both the training and testing sets to visually compare the model's performance.

## Forecasting
We created a function to predict the stock prices for the next 30 days after the testing period. We then plotted the forecasted prices to visualize the future trends.

## Libraries Used
- pandas
- matplotlib
- MinMaxScaler from sklearn.preprocessing
- Sequential, Dense, LSTM from tensorflow.keras.models and tensorflow.keras.layers
