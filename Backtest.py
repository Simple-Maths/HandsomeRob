# %%
from ShortStratergy import get_monthly_returns, run_backtest
import pandas as pd

leveraged = 'Files/SOXL.csv'
inverse = 'Files/SOXS.csv'

# leveraged = 'Files/JNUG.csv'
# inverse = 'Files/JDST.csv'
backtest =run_backtest(leveraged, inverse)

# %%
# --- IMPORTANT: Final Period Profit/Loss Capture ---
# This block of code is crucial for ensuring that the profit/loss accumulated
# during the *final, partial period* of the backtest is correctly included
# in the overall performance metrics.
if backtest.leveraged_etf_holdings != 0 or backtest.inverse_etf_holdings != 0:
    current_value = (backtest.leveraged_etf_holdings * data_leveraged['Close'].iloc[-1] +
                     backtest.inverse_etf_holdings * data_inverse['Close'].iloc[-1])
    profit_loss = current_value - (abs(backtest.leveraged_etf_holdings) * data_leveraged['Close'].iloc[backtest.last_trade_index] +
                                   abs(backtest.inverse_etf_holdings) * data_inverse['Close'].iloc[backtest.last_trade_index])
    backtest.trades_list.append({'Period End Date': data_leveraged.index[-1], 'Profit/Loss': profit_loss})

# %%
backtest['trades']['inverse']

# %%
lev_trades = pd.DataFrame(backtest['trades']['leveraged'])
inv_trades = pd.DataFrame(backtest['trades']['inverse'])

# %% [markdown]
# # View the trades

# %%
inv_trades

# %% [markdown]
# # Filter for different types of trades

# %%
lev_trades_rebalnce = lev_trades.query('trade_type != "initial" and trade_type !="additional"')

# %%
inv_trades_re =inv_trades.query('trade_type != "initial" and trade_type !="additional"')

# %%
inv_trades_re

# %%
lev_trades_rebalnce

# %% [markdown]
# # Calculate the average monthly return

# %%
import pandas as pd
from datetime import datetime

def calculate_avg_monthly_returns(df):
    """
    Calculate average monthly returns based on quarterly return percentages.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'date' and 'quarter_profit_loss_per' columns
    
    Returns:
    pd.DataFrame: Monthly returns with columns ['month', 'return']
    """
    # Convert date column to datetime if it's not already
    df['date'] = pd.to_datetime(df['date'])
    
    # Get only the rows where quarterly returns are calculated (non-zero)
    quarterly_returns = df[df['quarter_profit_loss_per'] != 0].copy()
    
    # Initialize dict to store results
    monthly_returns = {}
    
    # Process each quarterly return
    for i in range(len(quarterly_returns)):
        current_row = quarterly_returns.iloc[i]
        current_date = current_row['date']
        current_return = current_row['quarter_profit_loss_per']
        
        # If this is the first entry, we need to handle it specially
        if i == 0:
            # Get the number of months from the first trade to this quarterly return
            first_trade_date = df['date'].min()
            months_in_first_period = (
                (current_date.year - first_trade_date.year) * 12 + 
                current_date.month - first_trade_date.month + 1
            )
            monthly_return = current_return / months_in_first_period
            
            # Distribute this return across the months in the first period
            current_month = first_trade_date
            while current_month <= current_date:
                month_key = current_month.strftime('%Y-%m')
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []
                monthly_returns[month_key].append(monthly_return)
                current_month = (current_month + pd.DateOffset(months=1))
        else:
            # For subsequent quarters, we'll distribute the return across 3 months
            previous_date = quarterly_returns.iloc[i-1]['date']
            months_in_period = (
                (current_date.year - previous_date.year) * 12 + 
                current_date.month - previous_date.month + 1
            )
            monthly_return = current_return / months_in_period
            
            # Distribute this return across the months in this period
            current_month = previous_date + pd.DateOffset(months=1)
            while current_month <= current_date:
                month_key = current_month.strftime('%Y-%m')
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []
                monthly_returns[month_key].append(monthly_return)
                current_month = (current_month + pd.DateOffset(months=1))
    
    # Calculate average monthly returns
    result_data = []
    for month, returns in monthly_returns.items():
        result_data.append({
            'month': month,
            'return': sum(returns) / len(returns)
        })
    
    # Convert to DataFrame and sort by month
    result_df = pd.DataFrame(result_data)
    result_df = result_df.sort_values('month').reset_index(drop=True)
    
    return result_df

# %%
leveraged_monthly_returns = calculate_avg_monthly_returns(lev_trades)
inverse_monthly_returns = calculate_avg_monthly_returns(inv_trades)


# %%
leveraged_monthly_returns

# %%
inverse_monthly_returns

# %% [markdown]
# # Calculate the avg quarterly return for both the leveraged and inverse

# %%
def calculate_combined_returns(leveraged_trades, inverse_trades):
    """
    Combines two dataframes and calculates return percentage based on combined quarter_profit_loss.
    
    Parameters:
    leveraged_trades, inverse_trades (pd.DataFrame): Input dataframes with identical structure
    
    Returns:
    pd.DataFrame: DataFrame with date and return_per columns
    """
    # Concatenate the dataframes
    combined_df = pd.concat([leveraged_trades, inverse_trades], ignore_index=True)
    
    # Group by date and sum the quarter_profit_loss
    daily_profits = combined_df.groupby('date')['quarter_profit_loss'].sum()
    
    # Calculate return percentage
    # Creating a new dataframe with date and return_per
    result_df = pd.DataFrame({
        'date': daily_profits.index,
        'return_per_quarter': daily_profits / 2500
    }).reset_index(drop=True)
    
    # Sort by date
    result_df = result_df.sort_values('date')
    
    return result_df

# %%
combined_quarterly_returns = calculate_combined_returns(lev_trades_rebalnce, inv_trades_re)

# %%
combined_quarterly_returns

# %%


# %% [markdown]
# # Calculation of metrics

# %%
# Calculate key financial metrics

# 1. Cumulative Returns (starting with 1 to represent portfolio value)
portfolio_values = (1 + combined_quarterly_returns['return_per_quarter']).cumprod()
cumulative_returns = portfolio_values - 1

# 2. Total Return
total_return = cumulative_returns.iloc[-1]

# 3. Average Quarterly Return
avg_quarterly_return = combined_quarterly_returns['return_per_quarter'].mean()

# 4. Volatility (Standard Deviation)
volatility = combined_quarterly_returns['return_per_quarter'].std()

# 5. Maximum Drawdown (corrected calculation)
rolling_peak = portfolio_values.expanding().max()
drawdowns = (portfolio_values - rolling_peak) / rolling_peak  # Percentage decline from peak
max_drawdown = drawdowns.min()

# 6. Win Rate (percentage of positive quarters)
win_rate = (combined_quarterly_returns['return_per_quarter'] > 0).mean()

# 7. Best and Worst Quarters
best_quarter = combined_quarterly_returns['return_per_quarter'].max()
worst_quarter = combined_quarterly_returns['return_per_quarter'].min()

# Create a summary of metrics
metrics_summary = pd.Series({
    'Total Return': f'{total_return:.2%}',
    'Average Quarterly Return': f'{avg_quarterly_return:.2%}',
    'Quarterly Volatility': f'{volatility:.2%}',
    'Maximum Drawdown': f'{max_drawdown:.2%}',
    'Win Rate': f'{win_rate:.2%}',
    'Best Quarter': f'{best_quarter:.2%}',
    'Worst Quarter': f'{worst_quarter:.2%}'
})

print("Financial Metrics Summary:")
print(metrics_summary)


# %% [markdown]
# # Visulisation of results

# %%
# First let's check our data
print("Data types:")
print("Date dtype:", combined_quarterly_returns['date'].dtype)
print("Portfolio values dtype:", portfolio_values.dtype)
print("Drawdowns dtype:", drawdowns.dtype)

print("\nFirst few rows of data:")
print("Dates:", combined_quarterly_returns['date'].head())
print("Portfolio values:", portfolio_values.head())
print("Drawdowns:", drawdowns.head())

# Convert date to datetime if it isn't already
combined_quarterly_returns['date'] = pd.to_datetime(combined_quarterly_returns['date'])

# Create the visualization
import matplotlib.pyplot as plt

# Create figure and axes with a clean style
plt.rcParams['figure.figsize'] = [12, 10]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = 'white'

fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot portfolio value with explicit datetime conversion
ax1.plot(combined_quarterly_returns['date'], portfolio_values.values, 
         label='Portfolio Value ($)', color='#1f77b4', linewidth=2)
ax1.set_title('Portfolio Value Over Time (Starting with $1)', pad=20)
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Value')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot drawdowns with explicit datetime conversion
ax2.fill_between(combined_quarterly_returns['date'], drawdowns.values * 100, 0, 
                 color='red', alpha=0.3, label='Drawdowns')
ax2.set_title('Drawdowns Over Time', pad=20)
ax2.set_xlabel('Date')
ax2.set_ylabel('Drawdown (%)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Rotate x-axis labels for better readability
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Force display
plt.show()
print("\nPlot should be displayed above. If not, there might be an issue with the display backend.")



