# Stock Prediction Web Application

## Introduction

Welcome to the Stock Prediction Project! This is a web application that provides stock predictions for a given stock ticker. The project is developed using Python and utilizes various libraries for data mining, data preparation, and model creation. The core of the prediction model is built using Long Short-Term Memory (LSTM) neural networks, which have proven to be effective in time-series forecasting tasks. The web application is created with Streamlit, and the graphs are generated using Plotly and Matplotlib.

## Features

- **Stock Prediction:** Given a stock ticker, the application provides a prediction of the stock's future prices based on historical data.

- **Interactive Visualization:** The application displays interactive charts that allow users to explore historical stock prices and predictions.

- **User-Friendly Interface:** The web interface is designed to be intuitive and easy to use, making it accessible to both novice and experienced users.

## Installation

Follow these steps to set up the Stock Prediction Project on your local machine:

1. Clone the repository:

   ```bash
   git clone https://github.com/jv0619/Stock_Price_Prediction_System.git
   ```

2. Navigate to the project directory:

   ```bash
   cd stock-prediction-project
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```bash
   streamlit run stock_prediction_app.py
   ```

2. Access the application in your web browser by visiting `http://localhost:8501`.

3. In the application's user interface, enter a valid stock ticker symbol (e.g., AAPL for Apple Inc.) in the input field.

4. Click the "Predict" button to generate stock price predictions.

5. Explore the interactive charts displaying historical stock prices and predictions.

## Model and Data

- **LSTM Model:** The core of the stock prediction model is built using Long Short-Term Memory (LSTM) neural networks, which are capable of capturing sequential patterns in time-series data.

- **Data Sources:** The project utilizes data from financial data sources such as Yahoo Finance and Pandas Datareader to obtain historical stock price data.

- **Data Preparation:** Data preprocessing and feature engineering techniques are applied to prepare the data for model training. This includes handling missing values, scaling, and splitting the data into training and testing sets.

## Libraries Used

- **Data Mining and Preparation:**
  - `yfinance`: Fetch historical stock price data.
  - `pandas_datareader`: Access financial data sources.
  - `numpy`: Perform numerical operations.
  - `pandas`: Data manipulation and analysis.

- **Model Creation:**
  - `keras`: Build and train the LSTM model.
  - `sklearn`: Perform data preprocessing and evaluation of the model.

## Contributing

If you'd like to contribute to the project or report issues, please follow the guidelines outlined in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

Special thanks to the open-source community for developing and maintaining the libraries used in this project.

## Contact

If you have any questions or need further assistance, please feel free to contact [your-email@example.com].

Happy Stock Predicting!