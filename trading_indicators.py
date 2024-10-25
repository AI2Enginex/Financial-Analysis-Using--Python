import ta
import numpy as np
import pandas as pd
import ta.momentum
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
# Data Visualization using plotly
import plotly.express as px
import plotly.io as pio
from plotly.offline import init_notebook_mode, iplot
# pio.renderers.default = "vscode"
pio.renderers.default = "svg"
init_notebook_mode(connected=True)

class GetData:

    def __init__(self,symbol=None,start=None,end=None):

        self.df = yf.download(symbol + ".NS",start=start,end=end)


class MovingAverageCrossover(GetData):

    def __init__(self, symbol=None, start=None, end=None):
        super().__init__(symbol, start, end)
        self.ma_df = self.df.copy()

    # moving average crossover
    def moving_average_crossover(self, short_window=None, long_window=None, featurename=None,type=None):
        try:

            if type == 'simple':
                # Calculate short-term and long-term simple moving averages (eg. 7 vs 21, 50 vs 200)
                self.ma_df['MA_' + str(short_window)] = ta.trend.SMAIndicator(self.ma_df[featurename], window=short_window).sma_indicator()
                self.ma_df['MA_' + str(long_window)] = ta.trend.SMAIndicator(self.ma_df[featurename], window=long_window).sma_indicator()
            else:
                # Calculate short-term and long-term exponential moving averages (eg. 7 vs 21, 50 vs 200)
                self.ma_df['MA_' + str(short_window)] = ta.trend.EMAIndicator(self.ma_df[featurename], window=short_window).ema_indicator()
                self.ma_df['MA_' + str(long_window)] = ta.trend.EMAIndicator(self.ma_df[featurename], window=long_window).ema_indicator()


            # Generate signals
            self.ma_df['Signal'] = 0
            self.ma_df['Signal'] = [1 if self.ma_df['MA_' + str(short_window)][i] > self.ma_df['MA_' + str(long_window)][i] 
                                else -1 for i in range(len(self.ma_df))]

            self.ma_df['Position'] = self.ma_df['Signal'].diff()
            return self.ma_df
        
        except Exception as e:
            print(e)
        
    def plot_moving_average_crossover(self, featurename=None, short_window=None, long_window=None,ma_type=None):

        try:
            ma_result = self.moving_average_crossover(short_window=short_window,long_window=long_window,featurename=featurename,type=ma_type)
            fig = go.Figure()

            fig.add_trace(go.Candlestick(x=ma_result.index,
                                open=ma_result['Open'], high=ma_result['High'],
                                low=ma_result['Low'], close=ma_result['Close'],
                                name='OHLC'))

            # Add short-term moving average line
            fig.add_trace(go.Scatter(x=ma_result.index, y=ma_result['MA_' + str(short_window)],
                                    mode='lines', name=f'SMA {short_window}'))

            # Add long-term moving average line
            fig.add_trace(go.Scatter(x=ma_result.index, y=ma_result['MA_' + str(long_window)],
                                    mode='lines', name=f'SMA {long_window}'))

            # Add buy signals
            buy_signals = ma_result[ma_result['Position'] == 2]
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals[featurename],
                                    mode='markers', marker=dict(color='green', size=10),
                                    name='Buy Signal'))

            # Add sell signals
            sell_signals = ma_result[ma_result['Position'] == -2]
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals[featurename],
                                    mode='markers', marker=dict(color='red', size=10),
                                    name='Sell Signal'))

            # Add titles and labels
            fig.update_layout(title='Moving Average Crossover Strategy',
                            xaxis_title='Date',
                            yaxis_title=f'{featurename} Price',
                            legend_title='Legend',
                            xaxis_rangeslider_visible=False)

            # Show plot
            fig.show()
        except Exception as e:
            print("Error!!",e)
 
    
class RelativeStrengthIndex(GetData):

    def __init__(self, symbol=None, start=None, end=None):
        super().__init__(symbol, start, end)
        self.rsi_df = self.df.copy()

    # Relative Strength Index (RSI) Strategy
    def rsi_strategy(self, rsi_period=None, rsi_overbought=None, rsi_oversold=None,featurename=None):
        
        try:
            self.rsi_df['RSI'] = ta.momentum.RSIIndicator(self.rsi_df[featurename], window=rsi_period).rsi()
            # Buy when RSI < lower (oversold), Sell when RSI > upper (overbought)
            self.rsi_df['Signal'] = 0
            self.rsi_df['Signal'][rsi_period:] = [1 if self.rsi_df['RSI'][i] < rsi_oversold else (-1 if self.rsi_df['RSI'][i] > rsi_overbought else 0) for i in range(rsi_period, len(self.rsi_df))]
            
            self.rsi_df['Position'] = self.rsi_df['Signal'].diff()
            return self.rsi_df
        except Exception as e:
            return e
        
    def plot_rsi(self,period=None,upper=None,lower=None,featurename=None):

        try:
            rsi_result = self.rsi_strategy(rsi_period=period,rsi_overbought=upper,rsi_oversold=lower,featurename=featurename)
            # Plotly figure setup
            fig = go.Figure()

            # Add RSI line
            fig.add_trace(go.Scatter(x=rsi_result.index, y=rsi_result['RSI'],
                                    mode='lines', name='RSI', line=dict(color='blue')))

            # Add overbought and oversold lines
            fig.add_hline(y=upper, line=dict(color='red', dash='dash'), annotation_text="Overbought", annotation_position="bottom right")
            fig.add_hline(y=lower, line=dict(color='green', dash='dash'), annotation_text="Oversold", annotation_position="bottom right")

            # Add buy signals (when RSI crosses above lower)
            buy_signals = (rsi_result['RSI'] < lower) & (rsi_result['RSI'].shift(1) >= lower)  # Buy when previous RSI was >= lower
            fig.add_trace(go.Scatter(x=rsi_result.index[buy_signals],
                                    y=rsi_result['RSI'][buy_signals],
                                    mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                                    name='Buy Signal'))

            # Add sell signals (when RSI crosses below upper)
            sell_signals = (rsi_result['RSI'] > upper) & (rsi_result['RSI'].shift(1) <= upper)  # Sell when previous RSI was <= upper
            fig.add_trace(go.Scatter(x=rsi_result.index[sell_signals],
                                    y=rsi_result['RSI'][sell_signals],
                                    mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                                    name='Sell Signal'))

            # Add titles and labels
            fig.update_layout(title='Relative Strength Index (RSI) with Buy and Sell Signals',
                            xaxis_title='Date',
                            yaxis_title='RSI',
                            legend_title='Legend')

            # Show plot
            fig.show()
        except Exception as e:
            print(e)

class BollingerBands(GetData):

    def __init__(self, symbol=None, start=None, end=None):
        super().__init__(symbol, start, end)
        self.bollinger_df = self.df.copy()

    # Bollinger Bands Strategy
    def bollinger_bands(self, window=None, num_std_dev=None,featurename=None):
        try:
            # Calculate Bollinger Bands using the ta library
            bollinger = ta.volatility.BollingerBands(close=self.bollinger_df[featurename], window=window, window_dev=num_std_dev)
            self.bollinger_df['Middle Band'] = bollinger.bollinger_mavg()   # Middle band (SMA)
            self.bollinger_df['Upper Band'] = bollinger.bollinger_hband()   # Upper band
            self.bollinger_df['Lower Band'] = bollinger.bollinger_lband()   # Lower band

            # Generate signals:
            # Buy signal when price is below the lower band
            # Sell signal when price is above the upper band
            self.bollinger_df['Signal'] = np.where(self.bollinger_df[featurename] < self.bollinger_df['Lower Band'], 1,  # Buy signal
                                         np.where(self.bollinger_df[featurename] > self.bollinger_df['Upper Band'], -1, 0))  # Sell signal

            # Calculate position changes
            self.bollinger_df['Position'] = self.bollinger_df['Signal'].diff()

            return self.bollinger_df
        except Exception as e:
            return e
        
    def plot_bollinger_bands(self,window=None,std=None,featurename=None):
        try:

            self.bollinger_data = self.bollinger_bands(window=window,num_std_dev=std,featurename=featurename)
            # Plotly figure setup
            fig = go.Figure()

            # Add OHLC (candlestick) chart for price
            fig.add_trace(go.Candlestick(x=self.bollinger_data.index,
                                        open=self.bollinger_data['Open'], high=self.bollinger_data['High'],
                                        low=self.bollinger_data['Low'], close=self.bollinger_data['Close'],
                                        name='Price', increasing_line_color='green', decreasing_line_color='red'))

            # Add Bollinger Bands
            fig.add_trace(go.Scatter(x=self.bollinger_data.index, y=self.bollinger_data['Upper Band'],
                                    line=dict(color='red', dash='dash'), name='Upper Band'))
            fig.add_trace(go.Scatter(x=self.bollinger_data.index, y=self.bollinger_data['Middle Band'],
                                    line=dict(color='blue', dash='dash'), name='Middle Band (SMA)'))
            fig.add_trace(go.Scatter(x=self.bollinger_data.index, y=self.bollinger_data['Lower Band'],
                                    line=dict(color='green', dash='dash'), name='Lower Band'))

            # Add titles and labels
            fig.update_layout(title='Bollinger Bands',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            legend_title='Legend',
                            xaxis_rangeslider_visible=False)

            # Show plot
            fig.show()
          
        except Exception as e:
            print(e)


class MACDCrossOver(GetData):

    def __init__(self, symbol=None, start=None, end=None):
        super().__init__(symbol, start, end)
        self.mac_df = self.df.copy()

    # MACD Crossover Strategy
    def macd_crossover(self, fastperiod=None, slowperiod=None, signalperiod=None,featurename=None):
        try:
            macd = ta.trend.MACD(self.mac_df[featurename], window_slow=slowperiod, window_fast=fastperiod,window_sign=signalperiod)

            # Add MACD and Signal Line to the DataFrame
            self.mac_df['MACD'] = macd.macd()
            self.mac_df['Signal_Line'] = macd.macd_signal()
            self.mac_df['MACD Diff'] = macd.macd_diff()

            # Generate signals:
            # Buy signal (1) when MACD crosses above the signal line
            # Sell signal (-1) when MACD crosses below the signal line
            self.mac_df['Signal'] = np.where(self.mac_df['MACD'] > self.mac_df['Signal_Line'], 1, 
                                         np.where(self.mac_df['MACD'] < self.mac_df['Signal_Line'], -1, 0))

            # Calculate position changes
            self.mac_df['Position'] = self.mac_df['Signal'].diff()

            return self.mac_df
        
        except Exception as e:
            return e
    
    def plot_macd(self,fastperiod=None, slowperiod=None, signalperiod=None,featurename=None):

        try:

            macd_result = self.macd_crossover(fastperiod=fastperiod,slowperiod=slowperiod,signalperiod=signalperiod,featurename=featurename)
           # Create a Plotly figure
            fig = go.Figure()

            # Add MACD line
            fig.add_trace(go.Scatter(x=macd_result.index, y=macd_result['MACD'],
                                    mode='lines', name='MACD', line=dict(color='blue')))
            
            # Add Signal line
            fig.add_trace(go.Scatter(x=macd_result.index, y=macd_result['Signal_Line'],
                                    mode='lines', name='Signal Line', line=dict(color='orange')))
            
            # Add MACD histogram (MACD Diff)
            fig.add_trace(go.Bar(x=macd_result.index, y=macd_result['MACD Diff'],
                                name='MACD Histogram', marker_color='gray'))

            # Generate Buy signals (MACD crosses above the Signal line)
            buy_signals = (macd_result['MACD'] > macd_result['Signal_Line']) & (macd_result['MACD'].shift(1) <= macd_result['Signal_Line'].shift(1))
            fig.add_trace(go.Scatter(x=macd_result.index[buy_signals],
                                    y=macd_result['MACD'][buy_signals],
                                    mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                                    name='Buy Signal'))

            # Generate Sell signals (MACD crosses below the Signal line)
            sell_signals = (macd_result['MACD'] < macd_result['Signal_Line']) & (macd_result['MACD'].shift(1) >= macd_result['Signal_Line'].shift(1))
            fig.add_trace(go.Scatter(x=macd_result.index[sell_signals],
                                    y=macd_result['MACD'][sell_signals],
                                    mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                                    name='Sell Signal'))

            # Add chart titles and layout
            fig.update_layout(title='MACD Crossover Strategy with Buy and Sell Signals',
                            xaxis_title='Date',
                            yaxis_title='MACD Value',
                            legend_title='Legend')

            # Show the chart
            fig.show()
           
        except Exception as e:
            print(e)



if __name__ == '__main__':

   pass