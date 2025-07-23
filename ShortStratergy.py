"""
Backtesting script for a short-selling strategy on leveraged and inverse ETFs.

This script defines and executes a trading strategy using the Backtrader library.
The core of the script is the `HandsomeRob`, which maintains short positions
in a pair of assets (e.g., SOXL and SOXS), rebalancing them quarterly.

The script includes components for:
- Detailed trade recording (`TradeRecorder`).
- The trading strategy logic (`HandsomeRob`).
- Performance analysis on a monthly basis (`MonthlyAnalyzer`).
- Custom data feeds for handling specific CSV formats.
- Runner functions to configure and execute the backtest.
"""
import backtrader as bt
import pandas as pd
from datetime import datetime
import calendar
import datetime

# Note: The following CSV files are loaded into global DataFrames but are not used
# by the backtesting functions (`run_backtest`, `get_monthly_returns`).
# These functions load data from paths provided as arguments.

file_path = 'Files/SOXL.csv'
df_1 = pd.read_csv(file_path)


file_path = 'Files/SOXS.csv'
df_2 = pd.read_csv(file_path)
df_1['Date'] = pd.to_datetime(df_1['Date'], utc=True)
df_2['Date'] = pd.to_datetime(df_2['Date'], utc=True)


class TradeRecorder:
    """
    Manages and records all trading activities during a backtest.

    This class provides a structured way to log trades, categorize them by product type
    (e.g., 'leveraged', 'inverse') and trade type (e.g., 'initial', 'rebalance'),
    and maintain a chronological history of all transactions. It also calculates
    and tracks profit and loss on a quarterly basis for each product type.

    Attributes:
        trade_history (dict): A nested dictionary to store trades categorized by product
                              and trade type.
        inverse_quarter_profit_loss (float): A running total of profit and loss for
                                             the 'inverse' product within the current quarter.
        leveraged_quarter_profit_loss (float): A running total of profit and loss for
                                               the 'leveraged' product within the current quarter.
        all_trades_chronological (list): A list of all trades, sorted chronologically.
    """
    def __init__(self):
        """Initializes the TradeRecorder with empty trade histories."""
        # Dictionary to store trades for each product and type
        self.trade_history = {
            'leveraged': {
                'initial': [],
                'rebalance-BUY': [],
                'rebalance-SELL': [],
                'additional': []
            },
            'inverse': {
                'initial': [],
                'rebalance-BUY': [],
                'rebalance-SELL': [],
                'additional': []
            }
        }
        self.inverse_quarter_profit_loss = 0  # Running total for the quarter
        self.leveraged_quarter_profit_loss = 0 
        # Chronological list of all trades
        self.all_trades_chronological = []


    # def __init__(self):


    def record_trade(self, product_type, trade_type, date, price, size, portfolio_value_before, portfolio_value_after, is_buy=None, last_trade = None):
        """
        Records a single trade, updating both categorized and chronological histories.

        This method constructs a detailed record for each trade, including its financial
        impact on the quarterly profit and loss (P&L). The P&L is calculated based
        on cash flow: sells increase the P&L, and buys decrease it.

        Args:
            product_type (str): The type of product traded, either 'leveraged' or 'inverse'.
            trade_type (str): The category of the trade, e.g., 'initial', 'rebalance', 'additional'.
            date (datetime.date): The date of the trade.
            price (float): The execution price of the trade.
            size (float): The number of shares traded.
            portfolio_value_before (float): The value of the position before the trade.
            portfolio_value_after (float): The value of the position after the trade.
            is_buy (bool, optional): Indicates if the trade was a buy (True) or a sell (False).
                                     This is crucial for correct P&L calculation. Defaults to None.
            last_trade (bool, optional): Flag to indicate if this is the final closing trade
                                         of a position. Defaults to None.
        """
        
        if product_type == 'inverse':
            quarterly_profit_loss = self.inverse_quarter_profit_loss
            value_toDivide = 1500
        else:
            quarterly_profit_loss = self.leveraged_quarter_profit_loss
            value_toDivide = 1000

        # Modify trade_type to include buy/sell distinction for rebalancing
        if trade_type == 'rebalance' and is_buy is not None:
            trade_type = f'rebalance-{"BUY" if is_buy else "SELL"}'

        # Calculate profit/loss for each trade
        trade_value = abs(price * size)
        
        if last_trade:
            # For the last trade, the value is calculated differently to capture final P&L.
            trade_value = value_toDivide - portfolio_value_before
            #print(trade_value)
        
        # Determine trade direction for logging purposes.
        if trade_type == 'initial':
            DIR = 'INIT'
        elif trade_type == 'additional':
            DIR = 'SELL'
            # Update quarterly P&L based on cash flow. Note: `is_buy` must be provided
            # for 'additional' trades for this to be calculated correctly.
            if is_buy:
                quarterly_profit_loss -= trade_value
            else:
                quarterly_profit_loss += trade_value

        elif trade_type != 'initial':  # For 'rebalance' trades
            if is_buy:
                quarterly_profit_loss -= trade_value
            else:
                quarterly_profit_loss += trade_value

        if 'rebalance' in trade_type:
            per  = quarterly_profit_loss/value_toDivide
        else:
            per = 0
            
  
        # Compile all trade information into a dictionary.
        trade_info = {
            'date': date,
            'price': price,
            'size': size,
            'value': trade_value,
            'position_value_before': portfolio_value_before,
            'position_value_after': portfolio_value_after,
            'product_type': product_type,
            'trade_type': trade_type,
            'quarter_profit_loss': quarterly_profit_loss,
            'quarter_profit_loss_per': per,
            'direction': 'BUY' if is_buy else 'SELL-R' if is_buy is not None else DIR,
        }

        # Reset quarterly P&L after a rebalance.
        if 'rebalance' in trade_type:
            quarterly_profit_loss = 0  # Reset for next quarter/rebalance

        # Update the recorder's P&L state.
        if product_type == 'inverse':
            self.inverse_quarter_profit_loss = quarterly_profit_loss
        else:
            self.leveraged_quarter_profit_loss = quarterly_profit_loss 
        
        
        # Add the trade to the categorized history.
        self.trade_history[product_type][trade_type].append(trade_info)
        
        # Add the trade to the chronological list and keep it sorted.
        self.all_trades_chronological.append(trade_info)
        self.all_trades_chronological.sort(key=lambda x: x['date'])

    def trades(self, product_type=None, trade_type=None, sort_by='date', ascending=True):
        """
        Retrieves a filtered and sorted list of trades.

        Args:
            product_type (str, optional): Filter by 'leveraged' or 'inverse'.
                                          Defaults to None (all products).
            trade_type (str, optional): Filter by 'initial', 'rebalance-BUY', 'rebalance-SELL',
                                        'additional', or 'rebalance' (for both buy and sell).
                                        Defaults to None (all types).
            sort_by (str): The key to sort trades by (e.g., 'date', 'value'). Defaults to 'date'.
            ascending (bool): The sort order. Defaults to True.

        Returns:
            list: A list of trade dictionaries matching the filter criteria.
        
        Raises:
            ValueError: If an invalid `product_type` is provided.
        """
        if product_type and product_type not in self.trade_history:
            raise ValueError(f"Invalid product type: {product_type}. Must be 'leveraged' or 'inverse'")

        # If no filters are provided, return all trades in chronological order
        if product_type is None and trade_type is None and sort_by == 'date':
            return (self.all_trades_chronological if ascending 
                   else list(reversed(self.all_trades_chronological)))

        all_trades = []
        products_to_check = [product_type] if product_type else ['leveraged', 'inverse']
        
        # Handle the case where we want all rebalance trades
        if trade_type == 'rebalance':
            trade_types_to_check = ['rebalance-BUY', 'rebalance-SELL']
        else:
            trade_types_to_check = ([trade_type] if trade_type else 
                                  ['initial', 'rebalance-BUY', 'rebalance-SELL', 'additional'])
        
        for prod in products_to_check:
            for t_type in trade_types_to_check:
                if t_type in self.trade_history[prod]:
                    all_trades.extend(self.trade_history[prod][t_type])
        
        # Sort the collected trades
        if sort_by == 'date':
            all_trades.sort(key=lambda x: x['date'], reverse=not ascending)
        else:
            all_trades.sort(key=lambda x: x[sort_by], reverse=not ascending)
            
        return all_trades

    def get_all_trades(self):
        """
        Returns all recorded trades in chronological order.

        Returns:
            list: A list of all trade dictionaries, sorted by date.
        """
        return self.all_trades_chronological

    def trade_summary(self, product_type=None):
        """
        Generates a statistical summary of trades for a given product type.

        Args:
            product_type (str, optional): The product to summarize ('leveraged' or 'inverse').
                                          If None, summarizes all trades. Defaults to None.

        Returns:
            dict: A dictionary containing summary statistics, such as total trades,
                  total value, and counts of each trade type.
        """
        trades = self.trades(product_type=product_type)
        if not trades:
            return {}
            
        return {
            'total_trades': len(trades),
            'total_value': sum(t['value'] for t in trades),
            'average_size': sum(t['size'] for t in trades) / len(trades),
            'average_price': sum(t['price'] for t in trades) / len(trades),
            'trade_types': {
                'initial': len([t for t in trades if t['trade_type'] == 'initial']),
                'rebalance_buys': len([t for t in trades if t['trade_type'] == 'rebalance-BUY']),
                'rebalance_sells': len([t for t in trades if t['trade_type'] == 'rebalance-SELL']),
                'additional': len([t for t in trades if t['trade_type'] == 'additional'])
            }
        }
        
        
        
                

class HandsomeRob(bt.Strategy):
    """
    A quarterly rebalancing strategy that maintains short positions in two assets,
    typically a leveraged ETF and its inverse counterpart.

    The strategy aims to hold a fixed dollar value for each short position. It
    rebalances these positions at the beginning of each quarter (January, April,
    July, October). Additionally, it has a mechanism to increase a short position
    if its value drops significantly due to a favorable price movement (i.e., the
    asset price falls), effectively "topping up" the position to its original
    target value.

    All positions are closed out on the last day of the backtest.

    Parameters:
        leveraged_allocation (float): The target dollar value for the short position
                                      in the leveraged asset. Default is 1000.
        inverse_allocation (float): The target dollar value for the short position
                                    in the inverse asset. Default is 1500.
        additional_allocation (float): This parameter is not directly used in the current
                                       implementation, as the logic tops up to the
                                       original target value. Default is 0.10.
        price_decrease_threshold (float): The threshold for increasing a position.
                                          If the position value falls to this percentage
                                          of the target allocation (e.g., 90%), more
                                          shares are shorted to bring the value back
                                          to the target. Default is 0.90.
        rebalance_months (int): The number of months in a rebalancing period.
                                Default is 3 (quarterly). Note: The current logic is
                                hardcoded for quarterly rebalancing.
        initial_portfolio_value (float): The initial cash of the portfolio. This is
                                         used for some internal calculations but the
                                         main portfolio cash is set in Cerebro.
                                         Default is 1500.
    """
    params = (
        ('leveraged_allocation', 1000),    
        ('inverse_allocation', 1500),      
        ('additional_allocation', 0.10),      
        ('price_decrease_threshold', 0.90),   
        ('rebalance_months', 3),             
        ('initial_portfolio_value', 2500),
    )

    def __init__(self):
        """Initializes the strategy and its state variables."""
        self.trade_recorder = TradeRecorder()
        
        # State tracking for the leveraged asset (data0)
        self.leveraged_size = 0
        self.leveraged_entry_price = None
        self.leveraged_additional_taken = False
        self.leveraged_quater_profit_loss = 0

        # State tracking for the inverse asset (data1)
        self.inverse_size = 0
        self.inverse_entry_price = None
        self.inverse_additional_taken = False
        self.inverse_quarter_profit_loss = 0

        self.last_rebalance_month = None
        self.initial_positions_opened = False
        
        # Map data index to its target allocation value
        self.allocation_map = {
            0: self.params.leveraged_allocation,
            1: self.params.inverse_allocation
        }

    def get_current_position(self, data):
        """
        Gets the current absolute size of the position for a given data feed.

        Args:
            data (bt.DataBase): The data feed to check the position for.

        Returns:
            float: The absolute size of the current position (always positive).
        """
        return abs(self.getposition(data).size)

    def calculate_target_position(self, current_price, current_portfolio_size, allocation_percentage):
        """
        Calculates the number of shares needed to adjust the position to the target value.

        Note: The name is misleading. This function calculates the *difference* in shares
        required, not the absolute target size.

        Args:
            current_price (float): The current price of the asset.
            current_portfolio_size (float): The current value of the position.
            allocation_percentage (float): The target dollar value for the position.

        Returns:
            float: The number of shares to buy (negative) or sell (positive) to
                   reach the target allocation.
        """
        target_value = (allocation_percentage - current_portfolio_size)
        return (target_value / current_price)

    def update_position_tracking(self, data_idx, size=None, price=None, additional_taken=None):
        """
        Updates the internal state tracking for a given position.

        Args:
            data_idx (int): The index of the data feed (0 for leveraged, 1 for inverse).
            size (float, optional): The new position size. Defaults to None.
            price (float, optional): The new entry price. Defaults to None.
            additional_taken (bool, optional): Flag indicating if an additional
                                               position has been taken. Defaults to None.
        """
        if data_idx == 0:
            if size is not None:
                self.leveraged_size = size
            if price is not None:
                self.leveraged_entry_price = price
            if additional_taken is not None:
                self.leveraged_additional_taken = additional_taken
        else:
            if size is not None:
                self.inverse_size = size
            if price is not None:
                self.inverse_entry_price = price
            if additional_taken is not None:
                self.inverse_additional_taken = additional_taken

    def execute_single_trade(self, data, target_size, data_idx):
        """
        Executes a trade to adjust a position to its target size and records it.

        Args:
            data (bt.DataBase): The data feed for the asset to be traded.
            target_size (float): The number of shares to trade (positive to sell, negative to buy).
            data_idx (int): The index of the data feed (0 or 1).
        """
        current_size = self.get_current_position(data)
        size_difference = target_size 
        
        price=data.close[0]
        position_value = current_size*price
        if size_difference != 0:
                is_buy = size_difference < 0  # True if buying, False if selling
                
                if is_buy:
                    self.buy(data=data, size=abs(size_difference))
                else:
                    self.sell(data=data, size=abs(size_difference))
                    
                # Record the trade with buy/sell information for rebalancing
                product_type = 'leveraged' if data_idx == 0 else 'inverse'
                trade_type = 'initial' if not self.initial_positions_opened else 'rebalance'
                
                self.trade_recorder.record_trade(
                    product_type=product_type,
                    trade_type=trade_type,
                    date=self.data0.datetime.date(0),
                    price=data.close[0],
                    size=abs(size_difference),
                    portfolio_value_before=position_value,
                    portfolio_value_after=position_value+size_difference*price,
                    is_buy=is_buy if trade_type == 'rebalance' else None  # Only pass is_buy for rebalance trades
                )
        
        self.update_position_tracking(data_idx, size=target_size, price=data.close[0])

    def initialize_position(self, data, data_idx):
        """
        Opens the initial short position for a single product.

        Args:
            data (bt.DataBase): The data feed for the asset.
            data_idx (int): The index of the data feed.
        """
        target_size = self.calculate_target_position(
            data.close[0],
            0, # Initial portfolio size is zero
            self.allocation_map[data_idx]
        )
        self.execute_single_trade(data, target_size, data_idx)

    def handle_price_decrease(self, data, data_idx):
        """
        Increases the short position if the asset price drops, making the position profitable.

        This logic "tops up" the short position back to its target dollar value when its
        current value drops below a specified threshold (e.g., 90% of target) due to a
        favorable price decrease.

        Note: The `trade_recorder.record_trade` call within this method does not pass the
        `is_buy` parameter, which may lead to incorrect P&L tracking for these
        "additional" trades in the `TradeRecorder`.

        Args:
            data (bt.DataBase): The data feed for the asset.
            data_idx (int): The index of the data feed.
        """
        if data_idx==0:
            target_value  = self.params.leveraged_allocation
        else:
            target_value = self.params.inverse_allocation
            
        price = data.close[0]
        current_size = self.get_current_position(data)
        current_value = price * current_size
        
        # Check if the position's value has fallen below the threshold
        if current_value <= target_value * self.params.price_decrease_threshold:
            # Calculate shares needed to sell to return to the target allocation
            additional_size = self.calculate_target_position(
                data.close[0],
                current_value,
                self.allocation_map[data_idx] 
            )

            self.sell(data=data, size=additional_size)
            price = data.close[0]
            current_size_before = current_size
            product_type = 'leveraged' if data_idx == 0 else 'inverse'
            
            # Record this "additional" sell trade.
            self.trade_recorder.record_trade(
                product_type=product_type,
                trade_type='additional',
                date=self.data0.datetime.date(0),
                price=data.close[0],
                size=additional_size,
                portfolio_value_before=price * current_size_before,
                portfolio_value_after=price * (current_size_before + additional_size)
            )
            new_total_size = self.get_current_position(data)
            self.update_position_tracking(data_idx, size=new_total_size, additional_taken=False)

    def handle_rebalance(self, data, data_idx):
        """
        Rebalances a single position back to its target dollar allocation.

        This involves calculating the necessary shares to buy or sell to align the
        current position value with its target value specified in `allocation_map`.

        Args:
            data (bt.DataBase): The data feed for the asset.
            data_idx (int): The index of the data feed.
        """
        price = data.close[0]
        current_size = self.get_current_position(data)
        portfolio_value = price*current_size
        target_size = self.calculate_target_position(
            data.close[0],
            portfolio_value,
            self.allocation_map[data_idx]
        )
        self.execute_single_trade(data, target_size, data_idx)
        self.update_position_tracking(data_idx, additional_taken=False)

    def get_last_quarter_end_date(self, current_date):
        """
        Calculates the end date of the quarter preceding the given date.
        
        Args:
            current_date (datetime.date): The current date.

        Returns:
            datetime.date: The date of the last day of the previous quarter.
        """
        current_month = current_date.month
        year = current_date.year

        if current_month <= 3:
            # Previous quarter ended in December of the previous year.
            return datetime.date(year - 1, 12, 31)
        elif current_month <= 6:
            # Previous quarter ended in March.
            return datetime.date(year, 3, 31)
        elif current_month <= 9:
            # Previous quarter ended in June.
            return datetime.date(year, 6, 30)
        else: # current_month <= 12
            # Previous quarter ended in September.
            return datetime.date(year, 9, 30)

    def close_all_positions(self):
        """
        Closes all open positions and records the final trades.

        This method is intended to be called on the last day of the backtest.
        It closes any open short positions by buying them back.

        Note: There is a potential issue in the `record_trade` call where `is_buy`
        is hardcoded to `False`. For closing a short position (a buy transaction),
        `is_buy` should typically be `True` for correct P&L attribution based on cash flow.
        """
        current_date = self.data0.datetime.date(0)
        record_date = self.get_last_quarter_end_date(current_date)
        
        # Close leveraged position
        if self.leveraged_size != 0:
            current_size = self.get_current_position(self.data0)
            current_value = abs(self.data0.close[0] * current_size)
            self.close(data=self.data0)
            
            self.trade_recorder.record_trade(
                product_type='leveraged',
                trade_type='rebalance',
                date=record_date,  # Use quarter end date
                price=self.data0.close[0],
                size=current_size,
                portfolio_value_before=current_value,
                portfolio_value_after=0,
                is_buy=False,  # This should likely be True for closing a short.
                last_trade=True
            )
            self.leveraged_size = 0

        # Close inverse position
        if self.inverse_size != 0:
            current_size = self.get_current_position(self.data1)
            current_value = abs(self.data1.close[0] * current_size)
            self.close(data=self.data1)
            self.trade_recorder.record_trade(
                product_type='inverse',
                trade_type='rebalance',
                date=record_date,  # Use quarter end date
                price=self.data1.close[0],
                size=current_size,
                portfolio_value_before=current_value,
                portfolio_value_after=0,
                is_buy=False,  # This should likely be True for closing a short.
                last_trade=True
            )
            self.inverse_size = 0

    def next(self):
        """
        The main logic loop of the strategy, executed on each data bar.
        """
        current_date = self.data0.datetime.date(0)
        
        # Determine if this is the last bar of the backtest.
        try:
            next_date = self.data0.datetime.date(1)
            is_last_day = False
        except IndexError:
            next_date = None
            is_last_day = True
        
        # Initialize positions on the first day of the backtest.
        if not self.initial_positions_opened:
            self.initialize_position(self.data0, 0)
            self.initialize_position(self.data1, 1)
            self.initial_positions_opened = True
            self.last_rebalance_month = current_date.month
            return

        # On the last day, close all positions and exit.
        if is_last_day:
            self.close_all_positions()
            return
            
        # Standard daily logic:
        # 1. Check if positions should be increased due to price drops.
        self.handle_price_decrease(self.data0, 0)
        self.handle_price_decrease(self.data1, 1)
        
        # 2. Check for quarterly rebalancing at the start of Jan, Apr, Jul, Oct.
        if next_date is not None and next_date.month in [1, 4, 7, 10]:
            # Ensure we don't rebalance multiple times in the same rebalance month.
            months_since_rebalance = abs(next_date.month - self.last_rebalance_month)
            if months_since_rebalance > 1:
                self.handle_rebalance(self.data0, 0)
                self.handle_rebalance(self.data1, 1)
                self.last_rebalance_month = next_date.month
  
class MonthlyAnalyzer(bt.Analyzer):
    """
    A Backtrader analyzer to calculate and track monthly portfolio returns.
    """
    def __init__(self):
        """Initializes the analyzer's state."""
        super(MonthlyAnalyzer, self).__init__()
        self.monthly_returns = {}
        self.current_month = None
        self.month_start_value = 0
        
    def start(self):
        """Called at the beginning of the backtest to set initial values."""
        super(MonthlyAnalyzer, self).start()
        self.current_month = self.strategy.datas[0].datetime.date(0).month
        self.month_start_value = self.strategy.broker.getvalue()
        
    def next(self):
        """Called on each data bar to check for month changes."""
        super(MonthlyAnalyzer, self).next()
        current_date = self.strategy.datas[0].datetime.date(0)
        
        # If a new month has started, calculate and record the previous month's return.
        if current_date.month != self.current_month:
            end_value = self.strategy.broker.getvalue()
            monthly_return = (end_value - self.month_start_value) / self.month_start_value * 100
            
            # Store the return with a 'YYYY-MM' key.
            month_key = f"{current_date.year}-{self.current_month:02d}"
            self.monthly_returns[month_key] = {
                'return': monthly_return,
                'end_value': end_value
            }
            
            # Reset for the new month.
            self.current_month = current_date.month
            self.month_start_value = end_value
            
    def stop(self):
        """Called at the end of the backtest to calculate the final month's return."""
        super(MonthlyAnalyzer, self).stop()
        # Calculate return for the last partial or full month.
        end_value = self.strategy.broker.getvalue()
        monthly_return = (end_value - self.month_start_value) / self.month_start_value * 100
        
        last_date = self.strategy.datas[0].datetime.date(0)
        month_key = f"{last_date.year}-{self.current_month:02d}"
        self.monthly_returns[month_key] = {
            'return': monthly_return,
            'end_value': end_value
        }
        
    def get_analysis(self):
        """
        Returns the collected analysis results.

        Returns:
            dict: A dictionary containing monthly returns and summary statistics.
        """
        monthly_returns = self.monthly_returns
        if monthly_returns:
            average_monthly_return = sum(m['return'] for m in monthly_returns.values()) / len(monthly_returns)
        else:
            average_monthly_return = 0
        
        return {
            'monthly_returns': monthly_returns,
            'average_monthly_return': average_monthly_return,
            'best_month': max(monthly_returns.items(), key=lambda x: x[1]['return'], default=(None, None)),
            'worst_month': min(monthly_returns.items(), key=lambda x: x[1]['return'], default=(None, None))
        }

class NAVData(bt.feeds.PandasData):
    """
    Custom Backtrader data feed for CSV files where the primary price is in a 'NAV' column.
    It maps the 'NAV' column to all OHLC price fields.
    """
    params = (
        ('datetime', None),    # 'datetime' is the index of the pandas DataFrame
        ('open', 'NAV'),       # Map 'NAV' to 'open'
        ('high', 'NAV'),       # Map 'NAV' to 'high'
        ('low', 'NAV'),        # Map 'NAV' to 'low'
        ('close', 'NAV'),      # Map 'NAV' to 'close'
        ('volume', -1),        # No volume data
        ('openinterest', -1),  # No open interest data
    )    
    
    
    
class ADJClose(bt.feeds.PandasData):
    """
    Custom Backtrader data feed for CSV files where the primary price is in a 'Close' column.
    It maps the 'Close' column to all OHLC price fields.
    """
    params = (
        ('datetime', None),    # 'datetime' is the index of the pandas DataFrame
        ('open', 'Close'),     # Map 'Close' to 'open'
        ('high', 'Close'),     # Map 'Close' to 'high'
        ('low', 'Close'),      # Map 'Close' to 'low'
        ('close', 'Close'),    # Map 'Close' to 'close'
        ('volume', -1),        # No volume data
        ('openinterest', -1),  # No open interest data
    )    
    
    
    
def calculate_cumulative_returns(monthly_returns_list):
    """
    Calculates cumulative returns from a list of monthly returns.

    Args:
        monthly_returns_list (list): A list of dictionaries, where each dictionary
                                     contains a 'return' key with a percentage value.

    Returns:
        list: A list of cumulative returns, compounded monthly.
    """
    cumulative_returns = []
    cumulative = 1.0  # Start with a base of 1.0
    
    for monthly_return in monthly_returns_list:
        # Compound the return (e.g., 5% return is 1.05)
        cumulative *= (1 + monthly_return['return']/100.0)
        # Store the cumulative return as a percentage (e.g., 1.05 becomes 5.0)
        cumulative_returns.append((cumulative - 1) * 100)
        
    return cumulative_returns

def get_monthly_returns(leveraged_data_path, inverse_data_path, initial_cash=2500):
    """
    Runs the backtest and returns monthly performance analysis.

    Note: This function appears to be outdated or incomplete. It calculates monthly
    returns via the `MonthlyAnalyzer` but does not correctly process or return them.
    The `run_backtest` function is more up-to-date for retrieving trade data.

    Args:
        leveraged_data_path (str): File path to the leveraged asset's CSV data.
        inverse_data_path (str): File path to the inverse asset's CSV data.
        initial_cash (float): The starting cash for the backtest. Defaults to 1500.

    Returns:
        dict: A dictionary containing an empty list for 'monthly_returns'.
    """
    df_1 = pd.read_csv(leveraged_data_path)
    df_2 = pd.read_csv(inverse_data_path)
    #print(df_1.columns)
    if 'BOM' in inverse_data_path or 'Direxion' in inverse_data_path:
        DataClass = ADJClose
    else:
        DataClass = ADJClose
        df_1['Date'] = pd.to_datetime(df_1['Date'], utc=True)
        df_2['Date'] = pd.to_datetime(df_2['Date'], utc=True)
        # df_1['NAV'] = df_1['NAV'].round(2)
        # df_2['NAV'] = df_2['NAV'].round(2)

    

    df_1['Date'] = pd.to_datetime(df_1['Date'])
    df_1.set_index('Date', inplace=True)
    df_1 = df_1.sort_index()
    
    leveraged_data = DataClass(dataname=df_1)
    
    df_2['Date'] = pd.to_datetime(df_2['Date'])
    df_2.set_index('Date', inplace=True)
    df_2 = df_2.sort_index()
    
    inverse_data = DataClass(dataname=df_2)
    
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    cerebro.adddata(leveraged_data, name="Leveraged")
    cerebro.adddata(inverse_data, name="Inverse")
    cerebro.addstrategy(HandsomeRob)
    cerebro.addanalyzer(MonthlyAnalyzer, _name='monthly')
    cerebro.broker.setcash(initial_cash)
    
    # Run backtest
    results = cerebro.run()
    
    # Extract analysis, but note that it's not fully processed.
    monthly_analysis = results[0].analyzers.monthly.get_analysis()
    
    monthly_returns_list = []
    
    # This block appears to be incomplete as it sorts an empty list.
    monthly_returns_list.sort(key=lambda x: x['month'])
    
    analysis_results = {
        'monthly_returns': monthly_returns_list,
    }
    
    return analysis_results



         
def run_backtest(leveraged_data_path, inverse_data_path, initial_cash=2500):
    """
    Runs the backtest and returns the detailed trade history from the strategy.

    This function configures and executes the backtest using the `HandsomeRob`,
    then extracts and returns all recorded trades from the `TradeRecorder`.

    Args:
        leveraged_data_path (str): File path to the leveraged asset's CSV data.
        inverse_data_path (str): File path to the inverse asset's CSV data.
        initial_cash (float): The starting cash for the backtest. Defaults to 2500.

    Returns:
        dict: A dictionary containing the lists of all trades for both the 'leveraged'
              and 'inverse' products.
    """
    df_1 = pd.read_csv(leveraged_data_path)
    df_2 = pd.read_csv(inverse_data_path)
    df_1['Date'] = pd.to_datetime(df_1['Date'], utc=True)
    df_2['Date'] = pd.to_datetime(df_2['Date'], utc=True)

    # Use the ADJClose data feed for all inputs
    DataClass = ADJClose
    
    df_1['Date'] = pd.to_datetime(df_1['Date'])
    # Set Date as index for Backtrader data feed
    df_1.set_index('Date', inplace=True)
    df_1 = df_1.sort_index()
    
    leveraged_data = DataClass(dataname=df_1)
    
    df_2['Date'] = pd.to_datetime(df_2['Date'])
    # Set Date as index for Backtrader data feed
    df_2.set_index('Date', inplace=True)
    df_2 = df_2.sort_index()
    
    inverse_data = DataClass(dataname=df_2)
    
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    cerebro.adddata(leveraged_data, name="Leveraged")
    cerebro.adddata(inverse_data, name="Inverse")
    cerebro.addstrategy(HandsomeRob)
    cerebro.addanalyzer(MonthlyAnalyzer, _name='monthly')
    cerebro.broker.setcash(initial_cash)
    
    # Run backtest
    results = cerebro.run()
    strat = results[0]  # Get the strategy instance from the results

    # Create the results dictionary with the recorded trades.
    analysis_results = {
        'trades': {
            'leveraged': strat.trade_recorder.trades('leveraged'),
            'inverse': strat.trade_recorder.trades('inverse')
        }
    }
    
    return analysis_results