
import pandas as pd
from ShortStratergy import run_backtest
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

direxion = {
    "AI 2X": {
        "leveraged": "AIBU",
        "inverse": "AIBD"
    },
    "Energy 2X": {
        "leveraged": "ERX",
        "inverse": "ERY"
    },
    "Gold Miners 2X": {
        "leveraged": "NUGT",
        "inverse": "DUST"
    },
    "Junior Gold Miners 2X": {
        "leveraged": "JNUG",
        "inverse": "JDST"
    },
    "Oil 2X": {
        "leveraged": "GUSH",
        "inverse": "DRIP"
    },
    "Treasury 20+ 3X": {
        "leveraged": "TMF",
        "inverse": "TMV"
    },
    "Treasury 7-10 3X": {
        "leveraged": "TYD",
        "inverse": "TYO"
    },
    "China 3X": {
        "leveraged": "YINN",
        "inverse": "YANG"
    },
    "MSCI Emerging Markets 3X": {
        "leveraged": "EDC",
        "inverse": "EDZ"
    },
    "SP500 3X": {
        "leveraged": "SPXL",
        "inverse": "SPXS"
    },
    "Russel 3X": {
        "leveraged": "TNA",
        "inverse": "TZA"
    },
    "Dow Jones 3X": {
        "leveraged": "WEBL",
        "inverse": "WEBS"
    },
    "Financial Select 3X": {
        "leveraged": "FAS",
        "inverse": "FAZ"
    },
    "Real Estate 3X": {
        "leveraged": "DRN",
        "inverse": "DRV"
    },
    "S&P 500 High Beta 3X": {
        "leveraged": "HIBL",
        "inverse": "HIBS"
    },
    "Biotechnology 3X": {
        "leveraged": "LABU",
        "inverse": "LABD"
    },
    "Semiconductor 3X": {
        "leveraged": "SOXL",
        "inverse": "SOXS"
    },
    "Technology 3X": {
        "leveraged": "TECL",
        "inverse": "TECS"
    },
}

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
            # Avoid division by zero if months_in_first_period is 0 (shouldn't happen with +1)
            if months_in_first_period > 0:
                monthly_return = current_return / months_in_first_period
            else:
                monthly_return = current_return # Fallback, though ideally months_in_first_period is always > 0
            
            # Distribute this return across the months in the first period
            current_month = first_trade_date
            while current_month <= current_date:
                month_key = current_month.strftime('%Y-%m')
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []
                monthly_returns[month_key].append(monthly_return)
                current_month = (current_month + pd.DateOffset(months=1))
        else:
            # For subsequent quarters, we'll distribute the return across months in the period
            previous_date = quarterly_returns.iloc[i-1]['date']
            months_in_period = (
                (current_date.year - previous_date.year) * 12 + 
                current_date.month - previous_date.month
            )
            # Add 1 if the current date is in the same month as previous, or if previous was end of quarter and current is start of new quarter
            if current_date.month >= previous_date.month and current_date.year == previous_date.year:
                 months_in_period += 1 # Include the current month for calculation
            elif current_date.month < previous_date.month and current_date.year > previous_date.year:
                months_in_period += 1 # Across year boundary, include current month

            if months_in_period > 0:
                monthly_return = current_return / months_in_period
            else:
                monthly_return = current_return # Fallback

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
        'return_per_quarter': daily_profits / 2500 # Assuming initial_portfolio_value of 2500
    }).reset_index(drop=True)
    
    # Sort by date
    result_df = result_df.sort_values('date')
    
    return result_df

def calculate_metrics(combined_quarterly_returns):
    """
    Calculates key financial metrics from combined quarterly returns.
    """
    if combined_quarterly_returns.empty:
        return {}

    # 1. Cumulative Returns (starting with 1 to represent portfolio value)
    portfolio_values = (1 + combined_quarterly_returns['return_per_quarter']).cumprod()
    cumulative_returns = portfolio_values.iloc[-1] - 1 # Total cumulative return

    # 2. Total Return
    total_return = cumulative_returns

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
    metrics_summary = {
        'Total Return': total_return,
        'Average Quarterly Return': avg_quarterly_return,
        'Quarterly Volatility': volatility,
        'Maximum Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Best Quarter': best_quarter,
        'Worst Quarter': worst_quarter
    }
    return metrics_summary, portfolio_values, drawdowns

def plot_results(pair_name, combined_quarterly_returns, portfolio_values, drawdowns, output_dir):
    """
    Generates and saves plots for portfolio value and drawdowns.
    """
    if combined_quarterly_returns.empty:
        print(f"No data to plot for {pair_name}.")
        return

    # Convert date to datetime if it isn't already
    combined_quarterly_returns['date'] = pd.to_datetime(combined_quarterly_returns['date'])

    # Create the visualization
    plt.rcParams['figure.figsize'] = [12, 10]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = 'white'

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot portfolio value with explicit datetime conversion
    ax1.plot(combined_quarterly_returns['date'], portfolio_values.values, 
             label='Portfolio Value ($)', color='#1f77b4', linewidth=2)
    ax1.set_title(f'Portfolio Value Over Time (Starting with $1) for {pair_name}', pad=20)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot drawdowns with explicit datetime conversion
    ax2.fill_between(combined_quarterly_returns['date'], drawdowns.values * 100, 0, 
                     color='red', alpha=0.3, label='Drawdowns')
    ax2.set_title(f'Drawdowns Over Time for {pair_name}', pad=20)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f'{pair_name}_backtest_plot.png')
    plt.savefig(plot_path)
    plt.close(fig) # Close the figure to free memory

def create_reports(pair_name, leveraged_trades, inverse_trades, all_pair_metrics):
    """
    Creates reports for a given pair of leveraged and inverse ETFs.
    """
    # Create a directory for the reports if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Create a directory for the pair
    pair_dir = f'reports/{pair_name}'
    if not os.path.exists(pair_dir):
        os.makedirs(pair_dir)

    # Convert trades to DataFrames
    leveraged_trades_df = pd.DataFrame(leveraged_trades)
    inverse_trades_df = pd.DataFrame(inverse_trades)

    # Save the leveraged and inverse trades to CSV files (Detailed Report for each symbol)
    leveraged_trades_df.to_csv(f'{pair_dir}/leveraged_trades_detailed.csv', index=False)
    inverse_trades_df.to_csv(f'{pair_dir}/inverse_trades_detailed.csv', index=False)
    print(f"Saved detailed trade reports for {pair_name}.")

    # Calculate combined quarterly returns
    if not leveraged_trades_df.empty and not inverse_trades_df.empty:
        # Filter for rebalance trades to calculate combined returns accurately
        lev_trades_rebalance = leveraged_trades_df.query('trade_type.str.contains("rebalance")', engine='python')
        inv_trades_rebalance = inverse_trades_df.query('trade_type.str.contains("rebalance")', engine='python')

        if not lev_trades_rebalance.empty and not inv_trades_rebalance.empty:
            combined_quarterly_returns = calculate_combined_returns(lev_trades_rebalance, inv_trades_rebalance)
            
            # Calculate metrics
            metrics_summary, portfolio_values, drawdowns = calculate_metrics(combined_quarterly_returns)
            
            # Save metrics to a text file (Decent Report for each pair)
            metrics_file_path = f'{pair_dir}/pair_summary_report.txt'
            with open(metrics_file_path, 'w') as f:
                f.write(f"Summary Report for {pair_name} Pair\\n\\n")
                for metric, value in metrics_summary.items():
                    if isinstance(value, (float)):
                        if 'Return' in metric or 'Drawdown' in metric or 'Volatility' in metric:
                            f.write(f"{metric}: {value:.2%}\\n")
                        elif 'Win Rate' in metric:
                             f.write(f"{metric}: {value:.2%}\\n")
                        else:
                            f.write(f"{metric}: {value:.4f}\\n")
                    else:
                        f.write(f"{metric}: {value}\\n")
            print(f"Saved summary report for {pair_name}.")

            # Add to overall metrics for total report
            all_pair_metrics[pair_name] = metrics_summary

            # Plot results
            plot_results(pair_name, combined_quarterly_returns, portfolio_values, drawdowns, pair_dir)
            print(f"Saved plots for {pair_name}.")
        else:
            print(f"Not enough rebalance trade data to calculate combined returns for {pair_name}.")
    else:
        print(f"No trade data available for {pair_name} to generate combined reports.")

def run_full_backtest():
    """
    Runs the backtest for all the pairs in the direxion dictionary and generates reports.
    """
    all_pair_metrics = {} # To store metrics for the total report

    for pair_name, pair in direxion.items():
        print(f"Running backtest for {pair_name}...")
        leveraged_path = f"Direxion-ETNS/{pair['leveraged']}.csv"
        inverse_path = f"Direxion-ETNS/{pair['inverse']}.csv"

        try:
            backtest_results = run_backtest(leveraged_path, inverse_path)
            # Pass all_pair_metrics to accumulate results
            create_reports(pair_name, backtest_results['trades']['leveraged'], 
                           backtest_results['trades']['inverse'], all_pair_metrics)
        except FileNotFoundError as e:
            print(f"Could not run backtest for {pair_name}: {e}. Make sure files exist at {leveraged_path} and {inverse_path}")
        except Exception as e:
            print(f"An error occurred during backtest for {pair_name}: {e}")

    # Generate total report after all pairs are processed
    if all_pair_metrics:
        total_report_path = 'reports/total_backtest_summary.txt'
        with open(total_report_path, 'w') as f:
            f.write("Total Backtest Summary Across All Pairs\\n\\n")
            # Convert to DataFrame for easier aggregation if needed, or just iterate
            for pair_name, metrics in all_pair_metrics.items():
                f.write(f"--- {pair_name} ---\\n")
                for metric, value in metrics.items():
                    if isinstance(value, (float)):
                        if 'Return' in metric or 'Drawdown' in metric or 'Volatility' in metric:
                            f.write(f"{metric}: {value:.2%}\\n")
                        elif 'Win Rate' in metric:
                            f.write(f"{metric}: {value:.2%}\\n")
                        else:
                            f.write(f"{metric}: {value:.4f}\\n")
                    else:
                        f.write(f"{metric}: {value}\\n")
                f.write("\\n")
        print(f"Generated total backtest summary at {total_report_path}")
    else:
        print("No metrics collected to generate a total report.")


if __name__ == '__main__':
    run_full_backtest()
