# Plan of Action: Full Backtest

This document outlines the steps to perform a full backtest on all available leveraged and inverse ETF pairs.

## 1. Understand the Existing Code

- [X] Analyze `Backtest.py` to understand the backtesting logic for a single pair.
- [X] Analyze `Fullbacktest.py` to see the existing JSON object with all the pairs.
- [X] Analyze `ShortStratergy.py` to understand the core backtesting logic.

## 2. Implement the Full Backtest

- [X] Modify `Fullbacktest.py` to iterate through all the pairs from the JSON object.
- [X] For each pair, run the backtest using the logic from `ShortStratergy.py`.
- [X] The data source should be the `Direxion-ETNS` directory.
- [ ] Save a decent report for each symbol.
- [ ] Save a detailed report for each pair.
- [ ] Save a total report.

## 3. Execution and Logging

- [ ] Run the `Fullbacktest.py` script.
- [ ] Keep track of progress and any issues in this file.

## Thoughts and Observations

*   **Initial thoughts:** The `Backtest.py` script seems to contain all the necessary logic for a single pair. The main task will be to refactor this code into a reusable function and then call it for each pair in `Fullbacktest.py`. The reporting part will require careful handling of data aggregation and file I/O.
*   **Update:** I've modified `Fullbacktest.py` to run the backtests and save the individual trade logs. Next, I'll run the script.