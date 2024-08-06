import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
# import mplfinance as mpf
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


# 主函数
def main():
    st.title('股票回测框架')

    # 用户输入
    ticker = st.sidebar.text_input('股票代码', 'sh600519')
    start_date = st.sidebar.date_input('开始日期', value=pd.to_datetime('2019-01-01'))
    end_date = st.sidebar.date_input('结束日期', value=pd.to_datetime('2024-07-31'))

    short_window = st.sidebar.slider('短期移动平均窗口', min_value=5, max_value=50, value=5)
    long_window = st.sidebar.slider('长期移动平均窗口', min_value=10, max_value=100, value=20)

    initial_capital = st.number_input('初始资本', value=1000000)
    take_profit = st.number_input('止盈点（百分比）', value=0.3, format="%.2f")
    stop_loss = st.number_input('止损点（百分比）', value=-0.07, format="%.2f")

    st.write('长短期均值策略')
    if st.button('执行回测'):
        # 获取股票数据
        try:
            data = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        except Exception as e:
            st.error(f"无法获取股票数据: {e}")
            return

        # 应用策略
        results, sharpe_ratio = simple_moving_account_balance(data, short_window, long_window, initial_capital,
                                                              stop_loss=stop_loss)

        # 显示结果
        st.subheader('均线走势')
        st.line_chart(results[['close', 'ShortMA', 'LongMA']])

        st.subheader('K线走势')

        # 创建一个具有两个子图的布局，其中第一个子图占据大部分空间
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03,
                            row_width=[0.2, 0.7])  # 调整行的高度比例

        # 添加 K 线图
        fig.add_trace(go.Candlestick(x=results.index,
                                     open=results['open'],
                                     high=results['high'],
                                     low=results['low'],
                                     close=results['close'],
                                     name='K Line'), row=1, col=1)

        # 添加短期和长期移动平均线
        fig.add_trace(go.Scatter(x=results.index, y=results['ShortMA'], mode='lines', name='Short MA', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=results.index, y=results['LongMA'], mode='lines', name='Long MA', line=dict(color='red')), row=1, col=1)

        # 添加成交量图
        fig.add_trace(go.Bar(x=results.index, y=results['volume'], name='Volume', marker_color='dimgray'), row=2, col=1)

        # 更新图表的布局
        fig.update_layout(
            title_text=f'{ticker} K-Line Chart',
            xaxis_rangeslider_visible=False,  # 隐藏 K 线图下的范围滑块
            xaxis=dict(
                rangeslider=dict(
                    visible=False  # 隐藏成交量图下的范围滑块
                ),
                type='category'  # 设置 X 轴为类别类型，避免自动填充缺失日期
            ),
            height=600,  # 整体图表的高度
            width=900,  # 整体图表的宽度
            showlegend=True,  # 显示图例
            legend=dict(
                orientation="h",  # 水平布局
                yanchor="bottom",  # 图例底部对齐
                y=1.02,  # 图例位置
                xanchor="right",  # 图例右对齐
                x=1  # 图例位置
            )
        )

        # 显示 Plotly 图表
        st.plotly_chart(fig, use_container_width=True)

        # Display Sharpe ratio
        st.subheader('夏普比率')
        st.write(f"夏普比率: {sharpe_ratio:.2f}")


if __name__ == '__main__':
    main()
