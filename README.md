# Analysis of LOB Data and Development of a Short-Term Trading Strategy for S50 Futures
## File Structure
```
.
├── submission-folder/                    # Main application code folder 
   └── split_data.ipynb           # Jupyter Notebook for data splitting
         ├── in_sample.csv                  # In-sample data file
         ├── out_sample.csv                 # Out-of-sample data file
   ├── EDA.py                         # Exploratory Data Analysis code
   ├── Strategy_in_sample.py          # Trading strategy for in-sample data
   ├── Strategy_out_sample.py         # Trading strategy for out-of-sample data
   ├── Performance_Metrics_in_sample.py      # Performance metrics for in-sample data
   ├── Performance_Metrics_out_sample.py     # Performance metrics for out-of-sample data
   └── requirements.txt               # Required libraries for the project
   └── Summary Report of Analysis and Trading Strategy  #summarizing insights)

```
## Usage
Run the application with the following command:
```sh
streamlit run ตามด้วยชื่อไฟล์python.py
```
🔗 [Download Data and Documentation Here](https://outlinerbcc-erytbxre62w34kff2pztbb.streamlit.app/) 🚀📊
## Core Concept
My strategy focuses on utilizing high-frequency data from the Limit Order Book (LOB) along with trade data to exploit market inefficiencies at the microstructure level. Specifically, I target:

- **Order Imbalance:** Detecting discrepancies in the volumes on the bid and ask sides.
- **Liquidity Imbalance:** Recognizing situations where liquidity is unevenly distributed between bids and asks.
- **Spread Dynamics:** Capitalizing on short-term fluctuations in the bid-ask spread.

These inefficiencies create opportunities to enter and exit the market rapidly. Although each trade might yield a small profit, frequent trading can lead to significant overall returns.

## Process and Methodology

### 1. Exploratory Data Analysis (EDA)
- **Data Understanding:**  
  - Analyze key LOB data points such as the best bid (Bid1), best ask (Ask1), and the volume at various levels.
  - Examine the spread dynamics and volatility in the data.
- **Signal Generation:**  
  - Calculate average volumes on the bid and ask sides (Average Bids/Asks Volume) to detect imbalances.
  - Use these imbalances as trading signals.

### 2. Strategy Development
- **Signal-Based Entry/Exit Rules:**  
  - **Long Position:** Open when the imbalance signal is positive (green), indicating a stronger bid side.
  - **Short Position:** Open when the imbalance signal is negative (red), indicating a stronger ask side.
- **Risk Management:**  
  - Set a Stop Loss (SL) at an appropriate distance (e.g., 5 points) to limit downside risk.
  - Calculate the appropriate lot size based on risk per trade (e.g., 1% of capital).
- **Execution Considerations:**  
  - Ensure the strategy handles high-frequency data efficiently by avoiding simultaneous open and close orders in the same time unit.

### 3. Backtesting and Performance Evaluation
- **Historical Testing:**  
  - Backtest the strategy on in-sample historical data.
- **Performance Metrics:**  
  - Evaluate using measures such as Sharpe Ratio, Maximum Drawdown, Profit Factor, Win Rate, and Expectancy.

### 4. Risk Management
- **Transaction Costs:**  
  - Incorporate commissions, exchange fees, and slippage into the backtest to ensure realistic performance.
- **Position Sizing:**  
  - Adjust lot sizes based on the defined risk parameters (e.g., risking 1% of total capital per trade).
- **Stop Loss Management:**  
  - Use Stop Loss orders to limit adverse movements and protect capital.

## Rationale for Choosing This Strategy
- **High Liquidity & Detailed Data:**  
  S50 Futures markets provide high liquidity and detailed LOB data, which allow for precise timing when entering and exiting trades, even with small price differentials.
- **Exploitation of Microstructure Inefficiencies:**  
  The strategy takes advantage of market microstructure inefficiencies (order and liquidity imbalances) to capture short-term opportunities.
- **Risk Management:**  
  A robust risk management framework (including Stop Loss, proper lot sizing, and transaction cost integration) helps protect capital and reduce overall risk.
- **High-Frequency Advantage:**  
  By executing trades at a high frequency, even small profits per trade can accumulate to generate attractive overall returns.

## Exploiting Market Inefficiencies
The strategy leverages market microstructure inefficiencies, specifically:
- **Order Imbalance:**  
  When there is an excess of bid orders relative to ask orders (or vice versa), prices may deviate from their fair value.
- **Liquidity Imbalance:**  
  Uneven liquidity distribution can lead to temporary mispricings that the strategy can exploit.

By identifying and reacting to these signals, the strategy is designed to take advantage of rapid price movements in the market.

---

*This document summarizes the conceptual framework and methodology behind the trading strategy developed using LOB data. The approach combines detailed exploratory analysis, a signal-based trading system, rigorous backtesting, and comprehensive risk management to exploit short-term market inefficiencies effectively.*
