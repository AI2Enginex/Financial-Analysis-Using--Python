# Provides ways to work with large multidimensional arrays
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot
import numpy as np
# Allows for further data manipulation and analysis
import pandas as pd

import plotly.graph_objects as go
import yfinance as yf  # Reads stock data

import datetime as dt  # For defining dates

# Data Visualization using plotly
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"
pio.renderers
init_notebook_mode(connected=True)


# Function that gets a dataframe by providing a ticker and starting date
def read_dataframe(ticker, syear, smonth, sday, eyear, emonth, eday):

    # Defines the time periods to use
    try:
        start = dt.datetime(syear, smonth, sday)
        end = dt.datetime(eyear, emonth, eday)

        # return the dataframe
        return yf.download(ticker + ".NS", start, end).reset_index()
    except Exception as e:
        return e


def daily_return(dataframe, timeframe, feature1, feature2):
    """
    Calculate the daily return of a feature based on a specified timeframe.

    This function computes the percentage change of the specified feature over the specified timeframe.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing the data.
    timeframe (int): Number of periods to compute the daily return.
    feature1 (str): Name of the new feature where the daily return will be stored.
    feature2 (str): Name of the feature to calculate the daily return from.

    Returns:
    pandas.DataFrame: DataFrame with the daily return values appended under a new column.

    If an error occurs during calculation, return the error message.
    """
    try:
        # Calculate the percentage change of the specified feature over the specified timeframe
        dataframe[feature1] = dataframe[feature2].pct_change(timeframe)
        return dataframe
    except Exception as e:
        # If an error occurs during calculation, return the error message
        return e


def cumalitive_return(df, feature1, feature2):
    """

    Cumulative return represents the aggregate growth of an investment over a specified period,
    reflecting both capital appreciation and reinvested income. It is calculated by multiplying
    the individual periodic returns and adding 1, then taking the cumulative product of these values.
    The resulting figure indicates the overall performance of the investment from its starting point
    to the end of the period. Higher cumulative returns signify greater profitability, while negative
    cumulative returns indicate losses.
    """
    try:
        # Calculate cumulative returns using cumprod() function
        df[feature1] = (1 + df[feature2]).cumprod()
        return df
    except Exception as e:
        # If an error occurs during calculation, return the error message
        return e


def annualized_return(df, start, end, feature):
    """
    Annualized return measures the average rate of return per year, adjusted for compounding.
    It is calculated by taking the mean return over the period, then multiplying by the number
    of trading days in a year (e.g., 252 for daily data). The result is expressed as a percentage.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data.
    start (str): Start date of the date range (format: 'YYYY-MM-DD').
    end (str): End date of the date range (format: 'YYYY-MM-DD').
    feature (str): Name of the feature to calculate the return from.

    Returns:
    float: Annualized return percentage.

    If an error occurs during calculation, return the error message.
    """
    try:
        # Filter DataFrame based on the specified date range
        filtered = df[(df['Date'] >= start) & (df['Date'] <= end)]
        # Calculate the mean return over the period and annualize it (assuming 252 trading days in a year)
        return (filtered[feature].mean() * 252) * 100
    except Exception as e:
        # If an error occurs during calculation, return the error message
        return e


def volatility(dataframe, feature_name, bins):
    """
    Plot a histogram of the daily returns based on the specified feature.

    Volatility measures the dispersion of returns for a given security or market index.
    It is often represented by the standard deviation of returns. A histogram of daily returns
    provides insights into the distribution and variability of returns over the given period.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing the data.
    feature_name (str): Name of the feature representing daily returns.

    Returns:
    None

    If an error occurs during plotting, return the error message.
    """
    try:
        # Create a histogram plot of the daily returns
        fig = px.histogram(dataframe, x=feature_name,
                           title='Histogram of Daily Return', nbins=bins)
        # Display the plot
        fig.show()
    except Exception as e:
        # If an error occurs during plotting, return the error message
        return e


def simple_moving_average(dataframe, feature, windowsize):
    """
    SMA is a widely used technical indicator that smooths out price data by calculating
    the average of a specified number of past prices. It helps identify trends and reversals
    by filtering out short-term fluctuations.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing the data.
    feature (str): Name of the feature to calculate the SMA from.
    windowsize (int): Size of the moving window for the SMA calculation.

    Returns:
    pandas.DataFrame: DataFrame with the SMA values appended under a new column.

    If an error occurs during calculation, return the error message.
    """
    try:
        # Calculate the Simple Moving Average
        dataframe['SIMPLE_MA_' +
                  str(windowsize)] = dataframe[feature].rolling(windowsize).mean()
        return dataframe
    except Exception as e:
        # If an error occurs during calculation, return the error message
        return e


def exponential_moving_average(dataframe, feature, windowsize):
    """
    EMA is a type of moving average that places greater weight on more recent data points,
    making it more responsive to recent price changes. It helps identify trends and reversals
    by providing a smoother representation of price movements compared to SMA.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing the data.
    feature (str): Name of the feature to calculate the EMA from.
    windowsize (int): Size of the moving window for the EMA calculation.

    Returns:
    pandas.DataFrame: DataFrame with the EMA values appended under a new column.

    If an error occurs during calculation, return the error message.
    """
    try:
        # Calculate the Exponential Moving Average
        dataframe['EXPONENTIAL_MA_' + str(windowsize)] = dataframe[feature].ewm(
            span=windowsize, adjust=False, min_periods=windowsize).mean()
        return dataframe
    except Exception as e:
        # If an error occurs during calculation, return the error message
        return e


def plot_ohlc_ma(dataframe, feature_name, name, linecolor):
    """
    Plot OHLC data along with a moving average.

    This function creates a Plotly figure containing an OHLC (Open-High-Low-Close) chart
    and a line plot representing the moving average (MA) of a specified feature.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing the OHLC data and the feature for MA calculation.
    feature_name (str): Name of the feature representing the moving average.
    name (str): Name of the moving average trace.


    If an error occurs during plotting, return the error message.
    """
    try:
        # Create OHLC trace
        ohlc_trace = go.Ohlc(x=dataframe['Date'],
                             open=dataframe['Open'],
                             high=dataframe['High'],
                             low=dataframe['Low'],
                             close=dataframe['Close'],
                             name='OHLC')

        # Create moving average trace
        ma_trace = go.Scatter(x=dataframe['Date'],
                              y=dataframe[feature_name],
                              mode='lines',
                              name=name,
                              line=dict(color=linecolor))

        # Create figure
        fig = go.Figure([ohlc_trace, ma_trace])

        # Update layout to hide range slider on x-axis
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Show the plot
        fig.show()
    except Exception as e:
        # If an error occurs during plotting, return the error message
        return e


def ohlc_ma_chart(dataframe, ma_feature1, ma_feature2, color1, color2, name1, name2):
    """
    Comparing pairs of Moving Averages against the price movement of the stock.

    This function creates a Plotly figure containing a candlestick chart representing OHLC data
    and two line plots representing moving averages of different features.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing the OHLC data and the features for MA calculation.
    ma_feature1 (str): Name of the feature representing the first moving average.
    ma_feature2 (str): Name of the feature representing the second moving average.
    color1 (str): Color of the line for the first moving average.
    color2 (str): Color of the line for the second moving average.
    name1 (str): Name of the first moving average trace.
    name2 (str): Name of the second moving average trace.



    If an error occurs during plotting, return the error message.
    """
    try:
        # Create Candlestick trace
        candlestick = go.Candlestick(x=dataframe['Date'],
                                     open=dataframe['Open'],
                                     high=dataframe['High'],
                                     low=dataframe['Low'],
                                     close=dataframe['Close'],
                                     name='OHLC')
        
        # Create Moving Average traces
        ma1_trace = go.Scatter(
            x=dataframe['Date'], y=dataframe[ma_feature1], mode='lines', name=name1, line=dict(color=color1))
        ma2_trace = go.Scatter(
            x=dataframe['Date'], y=dataframe[ma_feature2], mode='lines', name=name2, line=dict(color=color2))

        # Create figure
        fig = go.Figure(data=[candlestick, ma1_trace, ma2_trace])

        # Update layout to hide range slider on x-axis
        fig.update_layout(xaxis_rangeslider_visible=False,yaxis=dict(title='Close'))

        # Show plot
        fig.show()
    except Exception as e:
        # If an error occurs during plotting, return the error message
        return e


def macd(dataframe, fast, slow, timeframe, feature):
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator for a given feature.

    MACD is a trend-following momentum indicator that shows the relationship between two
    exponential moving averages (EMAs) of a feature's price. It consists of three components:
    MACD line, signal line, and histogram.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing the data.
    fast (int): Number of periods for the fast EMA.
    slow (int): Number of periods for the slow EMA.
    timeframe (int): Number of periods for the signal line EMA.
    feature (str): Name of the feature to calculate the MACD from.

    Returns:
    pandas.DataFrame: DataFrame with MACD and signal values appended under new columns.

    If an error occurs during calculation, return the error message.
    """
    try:
        # Calculate the fast and slow EMAs
        fast_ema = dataframe[feature].ewm(span=fast, adjust=False).mean()
        slow_ema = dataframe[feature].ewm(span=slow, adjust=False).mean()

        # Calculate MACD line
        dataframe['macd'] = fast_ema - slow_ema

        # Calculate signal line
        dataframe['signal'] = dataframe['macd'].ewm(
            span=timeframe, adjust=False).mean()
        
        # calculate difference between macd and signal
        dataframe['Histogram'] = dataframe['macd'] - dataframe['signal']

        return dataframe
    except Exception as e:
        # If an error occurs during calculation, return the error message
        return e


def macd_chart(dataframe, fast, slow, timeframe, feature1, feature2, feature3):

    try:
        macd_df = macd(dataframe, fast, slow, timeframe, feature1)
        # OHLC chart
        ohlc_fig = go.Figure(data=go.Ohlc(x=dataframe['Date'],
                                          open=dataframe['Open'],
                                          high=dataframe['High'],
                                          low=dataframe['Low'],
                                          close=dataframe['Close']))

        ohlc_fig.update_layout(
            title='OHLC Chart', xaxis_rangeslider_visible=False,yaxis=dict(title=feature1))

        # MACD chart
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=macd_df['Date'], y=macd_df[feature2], mode='lines', name='MACD'))
        macd_fig.add_trace(go.Scatter(
            x=macd_df['Date'], y=macd_df[feature3], mode='lines', name='Signal'))
        macd_fig.add_trace(
            go.Bar(x=macd_df['Date'], y=macd_df['Histogram'], name='MACD DIFF'))

        macd_fig.update_layout(title='MACD Chart')

        # Show OHLC and MACD charts
        ohlc_fig.show()
        macd_fig.show()
    except Exception as e:
        return e


def rsi_index(dataframe, timeframe, upper, lower, feature_name):
    """

    RSI is a momentum oscillator that measures the speed and change of price movements.
    It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing the data.
    timeframe (int): Number of periods to compute the RSI over.
    upper (int): Upper limit for RSI level classification.
    lower (int): Lower limit for RSI level classification.
    feature_name (str): Name of the feature to calculate the RSI from.

    Returns:
    None

    If an error occurs during calculation, return the error message.
    """
    try:
        # Calculate price change
        dataframe['Price Change'] = dataframe[feature_name].diff()
        # Calculate positive and negative price changes
        dataframe['Positive Change'] = dataframe['Price Change'].apply(
            lambda x: x if x > 0 else 0)
        dataframe['Negative Change'] = dataframe['Price Change'].apply(
            lambda x: abs(x) if x < 0 else 0)

        # Calculate average gain and average loss over a specified timeframe
        dataframe['Average Gain'] = dataframe['Positive Change'].rolling(
            window=timeframe).mean()
        dataframe['Average Loss'] = dataframe['Negative Change'].rolling(
            window=timeframe).mean()

        # Calculate Relative Strength (RS)
        dataframe['RS'] = dataframe['Average Gain'] / dataframe['Average Loss']

        # Calculate Relative Strength Index (RSI)
        dataframe['RSI'] = 100 - (100 / (1 + dataframe['RS']))

        # Classify RSI levels based on upper and lower limits
        dataframe['RSI Level'] = 'Neutral'
        dataframe.loc[dataframe['RSI'] > upper, 'RSI Level'] = 'Overbought'
        dataframe.loc[dataframe['RSI'] < lower, 'RSI Level'] = 'Oversold'
        return dataframe

    except Exception as e:
        return e


def rsi_chart(dataframe, timeframe, upper, lower, feature_name):

    try:
        rsi_df = rsi_index(dataframe, timeframe, upper, lower, feature_name)
        # Create OHLC trace
        ohlc_trace = go.Ohlc(x=rsi_df['Date'],
                             open=rsi_df['Open'],
                             high=rsi_df['High'],
                             low=rsi_df['Low'],
                             close=rsi_df['Close'],
                             name='OHLC')

        # Create OHLC chart
        ohlc_fig = go.Figure(ohlc_trace)
        ohlc_fig.update_layout(
            title='OHLC Chart', xaxis_rangeslider_visible=False)

        # Show OHLC chart
        ohlc_fig.show()

        # Create RSI trace
        rsi_trace = px.line(data_frame=dataframe, x='Date',
                            y='RSI', labels='RSI')
        rsi_trace.add_hline(y=upper, opacity=0.5)
        rsi_trace.add_hline(y=lower, opacity=0.5)

        # Create RSI chart
        rsi_fig = go.Figure(rsi_trace)
        rsi_fig.update_layout(title='RSI Chart')

        # Show RSI chart
        rsi_fig.show()
    except Exception as e:
        return e


def simple_line_chart(dataframe, feature, color, name):

    try:
        # Create a line chart trace
        line_trace = go.Scatter(
            x=dataframe['Date'], y=dataframe[feature], mode='lines', name=name, line=dict(color=color))

        # Create layout
        layout = go.Layout(title=name, xaxis=dict(
            title='Date'), yaxis=dict(title=feature))

        # Create figure
        fig = go.Figure(line_trace, layout)

        # Show plot
        fig.show()
    except Exception as e:
        return e
