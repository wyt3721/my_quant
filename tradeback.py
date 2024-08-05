import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt


# Define a function to fetch stock data based on the exchange
def get_stock_data(ticker, start_date, end_date):
    data = ak.stock_zh_a_daily(symbol=ticker, adjust="qfq")

    data.index = pd.to_datetime(data["date"])
    data = data[start_date:end_date]
    return data


# Define a simple moving average strategy with stop loss
def simple_moving_account_balance(data, short_window, long_window, initial_capital, transaction_cost=0.001,
                                  stop_loss=-0.07):
    # Calculate short-term and long-term moving averages
    data['ShortMA'] = data['close'].rolling(window=short_window).mean()
    data['LongMA'] = data['close'].rolling(window=long_window).mean()

    # Create a signal column
    data['Signal'] = 0.0
    data['Signal'][short_window:] = np.where(data['ShortMA'][short_window:] > data['LongMA'][short_window:], 1.0, 0.0)

    # Generate trading orders
    data['Positions'] = data['Signal'].diff()

    # Initial capital
    capital = initial_capital
    shares_held = 0
    data['Capital'] = np.nan
    data['Buy_Signal'] = 0
    data['Sell_Signal'] = 0
    data['Stop_Loss_Signal'] = 0

    # Calculate daily returns
    data['Return'] = data['close'].pct_change()

    # Calculate strategy returns
    data['Strategy_Return'] = data['Return'] * data['Positions'].shift(1)

    # Calculate cumulative strategy returns
    data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()

    # Calculate highest capital point so far
    data['Highest_Capital'] = data['Capital'].cummax()

    # Calculate daily account balance
    for i in range(len(data)):
        if i == 0:
            data.at[data.index[i], 'Capital'] = initial_capital
        else:
            if data.at[data.index[i], 'Positions'] == 1:
                # Buy
                buy_price = data.at[data.index[i], 'close'] * (1 + transaction_cost)
                if capital >= buy_price:
                    shares_to_buy = int(capital / buy_price)
                    shares_held += shares_to_buy
                    capital -= shares_to_buy * buy_price
                    data.at[data.index[i], 'Buy_Signal'] = data.at[data.index[i], 'close']
            elif data.at[data.index[i], 'Positions'] == -1:
                # Sell
                sell_price = data.at[data.index[i], 'close'] * (1 - transaction_cost)
                if shares_held > 0:
                    shares_to_sell = min(shares_held,
                                         int(capital / (data.at[data.index[i - 1], 'close'] * (1 - transaction_cost))))
                    shares_held -= shares_to_sell
                    capital += shares_to_sell * sell_price
                    data.at[data.index[i], 'Sell_Signal'] = data.at[data.index[i], 'close']

            # Check for stop loss condition
            if data.at[data.index[i], 'Highest_Capital'] > 0 and (
                    1 - data.at[data.index[i], 'Capital'] / data.at[data.index[i], 'Highest_Capital']) >= stop_loss:
                # Sell all shares due to stop loss
                sell_price = data.at[data.index[i], 'close'] * (1 - transaction_cost)
                capital += shares_held * sell_price
                shares_held = 0
                data.at[data.index[i], 'Stop_Loss_Signal'] = data.at[data.index[i], 'close']

            # Account for the value of the shares held
            data.at[data.index[i], 'Capital'] = capital + shares_held * data.at[data.index[i], 'close']

    # Calculate Sharpe ratio
    # Assuming a risk-free rate of 0
    rf = 0.0
    sharpe_ratio = np.sqrt(252) * (data['Strategy_Return'].mean() - rf) / data['Strategy_Return'].std()

    return data, sharpe_ratio


# Main function for the Streamlit app
def main():
    st.title('股票回测演示系统')

    # User inputs
    ticker = st.sidebar.text_input('股票代码', 'sh600519')
    start_date = st.sidebar.date_input('开始日期', value=pd.to_datetime('2019-01-01'))
    end_date = st.sidebar.date_input('结束日期', value=pd.to_datetime('2024-07-31'))

    short_window = st.sidebar.slider('短期移动平均窗口', min_value=5, max_value=50, value=5)
    long_window = st.sidebar.slider('长期移动平均窗口', min_value=10, max_value=100, value=20)

    initial_capital = st.number_input('初始资本', value=1000000)
    stop_loss = st.number_input('止损点（百分比）', value=-0.07, format="%.2f")

    if st.button('执行回测'):
        # Fetch stock data
        try:
            data = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        except Exception as e:
            st.error(f"无法获取股票数据: {e}")
            return

        # Apply the strategy
        results, sharpe_ratio = simple_moving_account_balance(data, short_window, long_window, initial_capital,
                                                              stop_loss=stop_loss)

        # Display results
        st.subheader('均线走势')
        st.line_chart(results[['close', 'ShortMA', 'LongMA']])

        # Display account balance changes
        st.subheader('余额变化')

        # Plotting account balance with buy/sell signals using matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results.index, results['Capital'], label='Capital', color='blue')
        ax.scatter(results[results['Buy_Signal'] != 0].index, results[results['Buy_Signal'] != 0]['Capital'],
                   color='green', label='Buy Signal', marker='^')
        ax.scatter(results[results['Sell_Signal'] != 0].index, results[results['Sell_Signal'] != 0]['Capital'],
                   color='red', label='Sell Signal', marker='v')
        ax.scatter(results[results['Stop_Loss_Signal'] != 0].index,
                   results[results['Stop_Loss_Signal'] != 0]['Capital'], color='purple', label='Stop Loss', marker='x')
        ax.set_xlabel('Date')
        ax.set_ylabel('Capital')
        ax.set_title('Capital with Buy/Sell/Stop Loss Signals')
        ax.legend()
        st.pyplot(fig)

        # Display Sharpe ratio
        st.subheader('夏普比率')
        st.write(f"夏普比率: {sharpe_ratio:.2f}")


if __name__ == '__main__':
    main()
