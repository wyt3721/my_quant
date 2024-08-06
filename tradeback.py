import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import mplfinance as mpf
from PIL import Image
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

            # Record the number of shares held at each step
            data.at[data.index[i], 'Shares_Held'] = shares_held

    # Calculate Sharpe ratio
    # Assuming a risk-free rate of 0
    rf = 0.0
    sharpe_ratio = np.sqrt(252) * (data['Strategy_Return'].mean() - rf) / data['Strategy_Return'].std()

    return data, sharpe_ratio


# Main function for the Streamlit app
def main():
    st.title('股票回测框架')

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
        st.subheader('K线走势')

        # Plotting K-line chart with moving averages using mplfinance
        mpf.plot(results, type='candle', style='charles',
                 title=f'{ticker} K-Line Chart',
                 ylabel='Price',
                 volume=True,  # This will create a separate subplot for volume
                 mav=(short_window, long_window),
                 figratio=(12, 10),
                 figscale=1.5,
                 tight_layout=True,
                 show_nontrading=False,
                 savefig=dict(fname='kline_chart.png', dpi=300))

        # Load the K-line chart image and display it in Streamlit
        kline_chart = Image.open('kline_chart.png')
        st.image(kline_chart, caption='K-Line Chart')

        # Display account balance changes
        st.subheader('余额变化')

        # After plotting the K-line chart, before displaying the account balance changes
        st.subheader('持股数量变化')

        # Plot the number of shares held
        fig_shares, ax_shares = plt.subplots(figsize=(12, 6))
        ax_shares.plot(results.index, results['Shares_Held'], label='Shares Held')
        ax_shares.set_title('Shares Held Over Time')
        ax_shares.set_xlabel('Date')
        ax_shares.set_ylabel('Number of Shares')
        ax_shares.legend()
        ax_shares.grid(True)

        # Save and display the shares held plot
        plt.savefig('shares_held.png', dpi=300)
        st.pyplot(fig_shares)

        # Display account balance changes
        st.subheader('账户余额变化')

        # Plot the account balance
        fig_balance, ax_balance = plt.subplots(figsize=(12, 6))
        ax_balance.plot(results.index, results['Capital'], label='Account Balance')
        ax_balance.set_title('Account Balance Over Time')
        ax_balance.set_xlabel('Date')
        ax_balance.set_ylabel('Account Balance (RMB)')
        ax_balance.legend()
        ax_balance.grid(True)

        # Save and display the account balance plot
        plt.savefig('account_balance.png', dpi=300)
        st.pyplot(fig_balance)

        # Display Sharpe ratio
        st.subheader('夏普比率')
        st.write(f"夏普比率: {sharpe_ratio:.2f}")


if __name__ == '__main__':
    main()
