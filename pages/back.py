import streamlit as st
import backtrader as bt
from datetime import datetime
import akshare as ak
import pandas as pd
import matplotlib
matplotlib.use('Agg')


# 定义交易策略
class SimpleMovingAverageStrategy(bt.Strategy):
    params = (
        ('fast_period', 20),  # 快速移动平均线周期
        ('slow_period', 50),  # 慢速移动平均线周期
        ('printlog', False),  # 打印日志
    )

    def log(self, txt, dt=None):
        ''' Logging function for this strategy '''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.fast_period)
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.slow_period)

        # Indicators for the plotting show
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
                                            subplot=True)
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0], plot=False)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},'
                    f' Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                         f'Cost: {order.executed.value:.2f}, '
                         f'Comm: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log(f'Close, {self.dataclose[0]:.2f}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.sma_fast[0] > self.sma_slow[0]:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.sma_fast[0] < self.sma_slow[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')


# Streamlit 应用程序
def main():
    st.title("股票策略回测")
    stock_code = st.text_input("请输入股票代码", "sh600000")
    start_date = st.date_input("开始日期", datetime(2020, 1, 1))
    end_date = st.date_input("结束日期", datetime(2024, 8, 7))
    fast_period = st.number_input("快速移动平均线周期", min_value=1, value=20)
    slow_period = st.number_input("慢速移动平均线周期", min_value=1, value=50)
    initial_capital = st.number_input("初始资本", min_value=1000, value=100000)

    if st.button("执行回测"):
        # 将 start_date 和 end_date 转换为字符串格式
        # start_date_str = start_date.strftime('%Y-%m-%d')
        # end_date_str = end_date.strftime('%Y-%m-%d')

        # 加载数据
        df = ak.stock_zh_a_daily(symbol=stock_code, adjust="qfq")  # 获取前复权数据
        # st.write(df.head())
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]  # 使用字符串比较
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)  # 确保索引为 datetime 类型
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # 回测设置
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.addstrategy(SimpleMovingAverageStrategy, fast_period=fast_period, slow_period=slow_period)
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))

        # 执行回测
        cerebro.run()
        st.write('ok')

        # 绘制结果
        fig = cerebro.plot(style='candlestick', plotly=True, iplot=True,volume=False, numfigs=1)[0][0]
        st.pyplot(fig)


if __name__ == "__main__":
    main()