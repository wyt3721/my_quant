import pandas as pd
import akshare as ak
import quantstats as qs
import streamlit as st
import streamlit.components.v1
from io import StringIO
import backtrader as bt
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SimpleMovingAverageStrategy(bt.Strategy):
    params = (
        ('fast_period', 9),  # 快速移动平均线周期
        ('slow_period', 21),  # 慢速移动平均线周期
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

def load_data(stock_codes, start_date, end_date):
    dfs = []
    for code in stock_codes:
        df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        dfs.append(df)
    return dfs
def Output_report(target, base):
    # Generate Quantstats report
    # Specify a path to save the HTML report
    report_path = "backtest_report.html"
    qs.reports.html(target, base, output=report_path, title='回测结果')

    # Read the contents of the HTML file
    with open(report_path, 'r', encoding='utf-8') as f:
        html_report = f.read()

    # Use a StringIO object to store the HTML content
    report_io = StringIO(html_report)

    # Display the Quantstats report
    st.components.v1.html(report_io.read(), height=1200, scrolling=True)


def main():
    st.title("单只持有策略回报")
    st.info('参照沪深300指数')

    # User inputs
    stock_code = st.sidebar.text_input("请输入股票代码", "sh601398")
    start_date = st.sidebar.date_input("开始日期", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("结束日期", datetime(2024, 8, 7))
    fast_period = st.sidebar.number_input("快速移动平均线周期", min_value=1, value=9)
    slow_period = st.sidebar.number_input("慢速移动平均线周期", min_value=1, value=21)
    initial_capital = st.sidebar.number_input("初始资本", min_value=1000, value=100000)

    if st.button("执行回测"):
        try:
            # Load data
            df = ak.stock_zh_a_daily(symbol=stock_code, adjust="qfq")
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df.set_index('date', inplace=True)


            df.index = pd.to_datetime(df.index)
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Backtest setup
            cerebro = bt.Cerebro(stdstats=False)
            cerebro = bt.Cerebro()
            cerebro.addstrategy(SimpleMovingAverageStrategy, fast_period=fast_period, slow_period=slow_period)
            cerebro.broker.setcash(initial_capital)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.adddata(bt.feeds.PandasData(dataname=df))

            # Add a Simple Moving Average Analyzer
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='sma_analyzer', timeframe=bt.TimeFrame.Days)

            # Execute the backtest
            results = cerebro.run()
            strat = results[0]
            final_portfolio_value = cerebro.broker.getvalue()
            st.write(f'组合价值: {final_portfolio_value:.2f} CNY')

            # Extract the values from the indicators and remove NaN values
            fast_ma_values = strat.sma_fast.array[fast_period:]
            slow_ma_values = strat.sma_slow.array[slow_period:]

            # Convert the array.array to list or numpy array
            fast_ma_values = list(fast_ma_values)
            slow_ma_values = list(slow_ma_values)

            # Plotting
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.2, 0.7])
            fig.add_trace(
                go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1,
                col=1)
            fig.add_trace(go.Scatter(x=df.index[fast_period:], y=fast_ma_values, name='Fast MA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[slow_period:], y=slow_ma_values, name='Slow MA'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'), row=2, col=1)
            fig.update_layout(title_text=f'Stock: {stock_code}, Fast MA: {fast_period}, Slow MA: {slow_period}',
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig)

            # Extract the daily portfolio values from the analyzer
            daily_portfolio_values = strat.analyzers.sma_analyzer.get_analysis()

            # Convert the daily portfolio values to a Series
            daily_returns = pd.Series(daily_portfolio_values).pct_change().dropna()
            start_date = datetime.strftime(start_date, '%Y%m%d')
            end_date = datetime.strftime(end_date, '%Y%m%d')
            #
            df_300 = ak.stock_zh_index_daily_em(symbol="sh000300", start_date=start_date, end_date=end_date)
            df = ak.stock_zh_index_daily_em(symbol="sh601398", start_date=start_date, end_date=end_date)
            #
            df_merge = df.merge(df_300, how='inner', on='date')
            # # # 将合并数据表的日期列改为时间格式
            df_merge['date'] = pd.to_datetime(df_merge['date'], format='%Y-%m-%d')
            #
            # # # 目标策略收益率Series，pct_change的作用是根据价格计算收益率
            target = df_merge['close_x'].pct_change()
            # # # 将索引设置为日期列
            target.index = df_merge['date']
            # # # 基准策略收益率Series，计算方法相同

            base = df_merge['close_y'].pct_change()
            base.index = df_merge['date']

            Output_report(target, base)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
