import pandas as pd
from ShortStratergy import run_backtest
import os
from datetime import datetime
import numpy as np

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

def get_quarter_end_dates(start_date, end_date):
    """Generate quarter end dates between start_date and end_date"""
    quarter_ends = []
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Find the first quarter end after start_date
    year = current_date.year
    month = current_date.month
    
    # Determine which quarter we're in and get the next quarter end
    if month <= 3:
        next_quarter_end = pd.Timestamp(year, 3, 31)
    elif month <= 6:
        next_quarter_end = pd.Timestamp(year, 6, 30)
    elif month <= 9:
        next_quarter_end = pd.Timestamp(year, 9, 30)
    else:
        next_quarter_end = pd.Timestamp(year, 12, 31)
    
    # If the current date is already past this quarter end, move to next
    if current_date > next_quarter_end:
        if next_quarter_end.month == 12:
            next_quarter_end = pd.Timestamp(year + 1, 3, 31)
        elif next_quarter_end.month == 3:
            next_quarter_end = pd.Timestamp(year, 6, 30)
        elif next_quarter_end.month == 6:
            next_quarter_end = pd.Timestamp(year, 9, 30)
        else:
            next_quarter_end = pd.Timestamp(year, 12, 31)
    
    # Generate all quarter ends
    current_quarter_end = next_quarter_end
    while current_quarter_end <= end_date:
        quarter_ends.append(current_quarter_end)
        
        # Move to next quarter
        if current_quarter_end.month == 12:
            current_quarter_end = pd.Timestamp(current_quarter_end.year + 1, 3, 31)
        elif current_quarter_end.month == 3:
            current_quarter_end = pd.Timestamp(current_quarter_end.year, 6, 30)
        elif current_quarter_end.month == 6:
            current_quarter_end = pd.Timestamp(current_quarter_end.year, 9, 30)
        else:
            current_quarter_end = pd.Timestamp(current_quarter_end.year, 12, 31)
    
    return quarter_ends

def calculate_quarterly_returns_for_pair(leveraged_trades, inverse_trades):
    """
    Calculate quarterly returns for a pair of leveraged/inverse ETFs
    """
    # Convert to DataFrames if they're not already
    leveraged_df = pd.DataFrame(leveraged_trades) if not isinstance(leveraged_trades, pd.DataFrame) else leveraged_trades
    inverse_df = pd.DataFrame(inverse_trades) if not isinstance(inverse_trades, pd.DataFrame) else inverse_trades
    
    if leveraged_df.empty or inverse_df.empty:
        return pd.DataFrame()
    
    # Combine both DataFrames
    combined_df = pd.concat([leveraged_df, inverse_df], ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # Filter for rebalance trades only (quarterly returns)
    rebalance_trades = combined_df[combined_df['trade_type'].str.contains('rebalance', na=False)]
    
    if rebalance_trades.empty:
        return pd.DataFrame()
    
    # Group by date and sum the quarter_profit_loss
    daily_profits = rebalance_trades.groupby('date')['quarter_profit_loss'].sum().reset_index()
    
    # Calculate return percentage (assuming $2500 initial portfolio)
    daily_profits['return_pct'] = daily_profits['quarter_profit_loss'] / 2500
    
    # Format the dates and returns
    quarterly_returns = daily_profits.copy()
    quarterly_returns['date'] = quarterly_returns['date'].dt.strftime('%m/%d/%Y')
    quarterly_returns['return_pct'] = (quarterly_returns['return_pct'] * 100).round(2)
    
    return quarterly_returns[['date', 'return_pct']]

def generate_quarterly_returns_csv():
    """
    Generate a CSV file with quarterly returns for all Direxion pairs
    """
    print("Starting quarterly returns generation...")
    
    # Dictionary to store all results
    all_quarterly_data = {}
    
    # Process each pair
    for pair_name, pair in direxion.items():
        print(f"Processing {pair_name}...")
        
        leveraged_path = f"Direxion-ETNS/{pair['leveraged']}.csv"
        inverse_path = f"Direxion-ETNS/{pair['inverse']}.csv"
        
        try:
            # Run backtest
            backtest_results = run_backtest(leveraged_path, inverse_path)
            
            # Calculate quarterly returns
            quarterly_returns = calculate_quarterly_returns_for_pair(
                backtest_results['trades']['leveraged'], 
                backtest_results['trades']['inverse']
            )
            
            if not quarterly_returns.empty:
                # Create column name using the pair symbols
                column_name = f"{pair['leveraged']}/{pair['inverse']}"
                
                # Store the data
                for _, row in quarterly_returns.iterrows():
                    date = row['date']
                    return_pct = row['return_pct']
                    
                    if date not in all_quarterly_data:
                        all_quarterly_data[date] = {}
                    
                    all_quarterly_data[date][column_name] = f"{return_pct:.2f}%"
                
                print(f"✓ Successfully processed {pair_name}")
            else:
                print(f"⚠ No quarterly data found for {pair_name}")
                
        except FileNotFoundError as e:
            print(f"✗ Could not find files for {pair_name}: {e}")
        except Exception as e:
            print(f"✗ Error processing {pair_name}: {e}")
    
    if not all_quarterly_data:
        print("No data collected. Please check your file paths and data.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_quarterly_data, orient='index')
    
    # Sort by date
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Convert index back to string format
    df.index = df.index.strftime('%m/%d/%Y')
    
    # Reset index to make date a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Fill NaN values with empty strings for cleaner output
    df = df.fillna('')
    
    # Save to CSV
    output_filename = 'quarterly_returns_summary.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"\n✓ Quarterly returns CSV generated: {output_filename}")
    print(f"📊 Data includes {len(df)} quarters across {len(df.columns)-1} ETF pairs")
    
    # Display sample of the data
    print("\nSample of generated data:")
    print(df.head(10).to_string(index=False))
    
    return df

if __name__ == '__main__':
    generate_quarterly_returns_csv()