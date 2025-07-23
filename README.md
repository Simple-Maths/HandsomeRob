# Handsome Rob: Leveraged ETF Shorting Strategy Backtest

This project provides a comprehensive backtesting framework for the "Handsome Rob" trading strategy, implemented using the `backtrader` library in Python. The strategy focuses on simultaneously short-selling a pair of correlated assets, typically a leveraged ETF and its inverse counterpart (e.g., SOXL and SOXS).All data was sourced from yahoo finance using the python library.

## Table of Contents
- [Strategy Explained](#strategy-explained)
  - [Core Concept](#core-concept)
  - [Quarterly Rebalancing](#quarterly-rebalancing)
  - [Position Top-Up Mechanism](#position-top-up-mechanism)
- [Project Structure](#project-structure)
- [Technical Components](#technical-components)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Files](#data-files)
  - [Running the Backtest](#running-the-backtest)
- [Analysis](#analysis)
- [Dependencies](#dependencies)

---

## Strategy Explained

The "Handsome Rob" strategy is built on the idea of holding simultaneous short positions in a leveraged asset and its inverse. The goal is to capitalize on the value decay inherent in many leveraged financial products, while maintaining a market-neutral stance.

### Core Concept
The strategy identifies a pair of assets (e.g., a 3x leveraged bull ETF and a 3x leveraged bear ETF on the same underlying index). It then opens short positions in both, with specific dollar-value allocations for each.

- **Leveraged Asset Allocation:** The target short position value for the leveraged asset is set to $1,000 (e.g., SOXL).
- **Inverse Asset Allocation:** The target short position value for the inverse asset is set to $1,500 (e.g., SOXS).

These allocations are defined as parameters within the strategy and remain constant targets throughout the backtest.

### Quarterly Rebalancing
At the beginning of each quarter (January, April, July, and October), the strategy rebalances both positions. The goal is to reset the value of each short position back to its original target allocation.

- If a position's value has increased (due to an adverse price movement), the strategy will buy back some shares to reduce the position's value to the target.
- If a position's value has decreased (due to a favorable price movement), the strategy will sell more shares to increase the position's value back to the target.

### Position Top-Up Mechanism
A key feature of the strategy is its response to favorable price movements within a quarter. If the price of an asset drops enough that the value of the short position falls below a certain threshold (e.g., 90% of its target allocation), the strategy intervenes without waiting for the quarterly rebalance.

It will short-sell additional shares to bring the position's value back up to its initial target allocation. This allows the strategy to capitalize on significant downward price trends proactively.

---

## Project Structure
The repository is organized as follows:

- **`ShortStratergy.py`**: The core of the project. This script contains the full implementation of the backtesting logic, including the strategy, trade recorder, and performance analyzers.
- **`HandsomeRobBacketest`**: A Jupyter Notebook designed to run the backtest and perform detailed post-trade analysis on the results. It provides a practical example of how to use the `ShortStratergy.py` script and visualize its output.
- **`requirements.txt`**: A list of all Python dependencies required to run the project.
- **`Files/`**: This directory is intended to store the historical price data for the assets being tested. The data should be in CSV format.
- **`README.md`**: This file.

---

## Technical Components
The `ShortStratergy.py` script is built from several key classes:

- **`TradeRecorder`**: A utility class designed to log every trade executed by the strategy. It categorizes trades as 'initial', 'rebalance', or 'additional' and tracks profit and loss on a quarterly basis.
- **`HandsomeRob(bt.Strategy)`**: The main strategy class that inherits from `backtrader.Strategy`. It contains all the logic for initialization, rebalancing, and the position top-up mechanism.
- **`MonthlyAnalyzer(bt.Analyzer)`**: A custom `backtrader` analyzer used to calculate month-over-month portfolio returns throughout the backtest period.
- **`run_backtest(...)`**: The primary function that configures and executes the backtest. It sets up the `backtrader` engine (`Cerebro`), adds the data feeds and the strategy, runs the backtest, and returns the results.
- **Custom Data Feeds (`NAVData`, `ADJClose`)**: Custom data feed classes that adapt pandas DataFrames with specific column names ('NAV' or 'Close') for use with `backtrader`.

---

## Getting Started

Follow these steps to set up and run the backtest on your local machine.

### Prerequisites
- Python 3.6 or higher
- `pip` package manager
- `virtualenv` (recommended)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Files
The backtest requires historical price data in CSV format.
1.  Place your CSV files inside the `Files/` directory.
2.  The CSV file must contain at least a `Date` column and a price column (e.g., `Close` or `NAV`). The date format should be recognizable by pandas (e.g., `YYYY-MM-DD`).

**Example `SOXL.csv`:**
```csv
Date,Open,High,Low,Close,Adj Close,Volume
2021-01-04,25.0,25.2,24.5,25.1,25.1,100000
...
```

### Running the Backtest
The easiest way to run the backtest and analyze the results is by using the **`HandsomeRobBacketest.ipynb`** Jupyter Notebook.

1.  **Start Jupyter:**
    ```bash
    jupyter notebook
    ```
2.  Open `HandsomeRobBacketest.ipynb`.
3.  In the first cell, ensure the `leveraged` and `inverse` variables point to the correct CSV files in the `Files/` directory.
    ```python
    leveraged = 'Files/SOXL.csv'
    inverse = 'Files/SOXS.csv'
    ```
4.  Run the cells in the notebook sequentially to execute the backtest and see the analysis.

---

## Analysis
The Jupyter Notebook provides a comprehensive analysis of the backtest results. The `run_backtest` function returns a dictionary containing detailed trade logs and performance metrics.

The notebook demonstrates how to:
- View individual trades for both the leveraged and inverse assets.
- Separate trades by type (initial, rebalance, additional).
- Calculate monthly returns for each asset based on the recorded quarterly P&L.
- Calculate combined quarterly returns for the entire portfolio.
- Compute key performance metrics such as:
  - **Cumulative Returns**
  - **Total Return**
  - **Volatility** (Standard Deviation of returns)
  - **Maximum Drawdown**

---

## Dependencies
This project relies on the following major Python libraries:

- **`backtrader`**: For event-driven backtesting of trading strategies.
- **`pandas`**: For data manipulation and analysis.
- **`numpy`**: For numerical operations.
- **`matplotlib`**: For plotting and data visualization (used by `backtrader`).
- **`seaborn`**: For enhanced data visualization. 
