import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Financial Data Analysis and Trading Strategy Backtest", layout="wide")

# =============================================================================
# Explanatory Text (README-style)
# =============================================================================
st.title("Financial Data Analysis and Trading Strategy Backtest")

st.markdown("""
# Explanation of the Code and Trading Strategy

---

## Code Structure

### 1. Data Loading and Preprocessing  
- **Reading the CSV File:**  
  The code starts by reading data from the `out_sample.csv` file.
- **Creating the Datetime Column:**  
  The `Date` and `Time_` columns are concatenated and converted into a datetime object using `pd.to_datetime`.
- **Sorting and Setting the Index:**  
  The data is then sorted by time and the `Datetime` column is set as the index to support time series analysis.

### 2. Calculating Strategy-Related Variables  
- **Average Bids/Asks Volume:**  
  The average volume is calculated for the bid side (vBid1–vBid5) and the ask side (vAsk1–vAsk5).
- **Difference (Imbalance):**  
  The `difference` is computed as `average_asks_volume - average_bids_volume`.
  - If `difference >= 0`, the signal is considered “green” (suggesting a potential long signal).
  - If `difference < 0`, the signal is considered “red” (suggesting a potential short signal).

### 3. Time Range Selection  
- The user can select a start and end date via Streamlit’s sidebar, and the data is filtered to include only the chosen time range.

### 4. Creating Background Shapes  
- The code examines the `difference` value row by row to segment periods with a consistent signal (green or red).
- For each segment with a consistent signal, a rectangle (shape) is created spanning the entire y-axis (from 0 to 1 in “paper” coordinates) using green for positive signals and red for negative signals.

### 5. Plotting the Bid1/Ask1 Chart with Background  
- A Plotly chart is generated showing the time series for **Bid1** (blue) and **Ask1** (red).  
- The layout is enhanced by overlaying the background shapes to visually indicate periods of market imbalance.

### 6. Backtest Simulation  
- **Key Parameters:**  
  - **Contract Size:** 200 Baht per point  
  - **Stop Loss (SL):** 5 points  
  - **Commission Fee:** 40 Baht per trade (combined for entry and exit)  
  - **Lot Size:** 1 contract  
- **Trading Rules:**  
  - When no position is open (`position == 0`), the system opens a new position based on the signal:
    - If `difference >= 0` (green signal), a Long position is opened at the `Ask1` price.
    - If `difference < 0` (red signal), a Short position is opened at the `Bid1` price.
  - When a position is already open (`position != 0`), the system checks for exit conditions:
    - **For Long:**  
      - If `Bid1` falls to or below `entry_price - 5` points, the Stop Loss is triggered and the Long is closed.
      - If the signal reverses to red (i.e., no longer green), the Long is closed at the `Bid1` price.
    - **For Short:**  
      - If `Ask1` rises to or above `entry_price + 5` points, the Stop Loss is triggered and the Short is closed.
      - If the signal reverses to green, the Short is closed at the `Ask1` price.
- **Preventing Actions in the Same Timestamp:**  
  A variable (`last_trade_timestamp`) is used to store the timestamp of the most recent trade. If a new row has the same timestamp, it is skipped to avoid multiple actions in one time unit.

### 7. Trade Log Recording and Display  
- When a trade is closed, its details (Entry Time, Exit Time, Entry Price, Exit Price, Profit, and Remark) are recorded in a DataFrame (`trades_df`).
- The total net profit is calculated and displayed.

### 8. Plotting Entry/Exit Points on the Price Chart  
- **Color and Marker Differentiation:**  
  - **Long Entry:** Displayed with a green circle.  
  - **Long Exit:** Displayed with a green cross.  
  - **Short Entry:** Displayed with a dark red circle.  
  - **Short Exit:** Displayed with a dark red cross.
- These markers are plotted on the Bid1/Ask1 chart so that it’s clear where positions were entered and exited.

---

## Strategy Advantages

1. **Exploiting Market Microstructure Inefficiencies:**  
   - This strategy leverages high-frequency LOB data to detect imbalances in order volume (Order Imbalance) and liquidity (Liquidity Imbalance) in the market.  
   - The computed `difference` provides a clear signal indicating short-term price trends.

2. **High-Frequency Trading Approach:**  
   - The “One Action per Row” rule, along with measures to prevent multiple actions in the same timestamp, enables the system to quickly react to rapid market changes.  
   - The defined Stop Loss helps limit losses when the market moves unfavorably.

3. **Risk Management:**  
   - A strict 5-point Stop Loss and clear commission calculations are integral to controlling risk.  
   - The Lot Size is determined based on the capital at risk (here, 1 contract), ensuring effective capital management.

4. **Visualization and Analysis:**  
   - The charts display both the raw price data (Bid1 and Ask1) with background colors indicating market imbalance and clearly marked entry/exit points.  
   - This comprehensive visualization aids in understanding trade execution and refining the strategy based on backtest results.

---

**In summary,** this strategy is well-suited for trading in the S50 Futures market as it effectively captures microstructure inefficiencies.
""")

# =============================================================================
# 1. Load and Preprocess Data
# =============================================================================
st.markdown("### 1. Load Data from CSV")
csv_file = 'out_sample.csv'
df = pd.read_csv(csv_file)

# Create Datetime column and set as index
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time_'])
df.sort_values('Datetime', inplace=True)
df.set_index('Datetime', inplace=True)

st.subheader("Sample Data:")
st.dataframe(df.head())

# =============================================================================
# 2. Compute Derived Variables: MidPrice, Spread, and Imbalance
# =============================================================================
st.markdown("### 2. Compute Derived Variables: MidPrice, Spread, and Imbalance")
df['MidPrice'] = (df['Bid1'] + df['Ask1']) / 2
df['Spread'] = df['Ask1'] - df['Bid1']
df['average_bids_volume'] = (df['vBid1'] + df['vBid2'] + df['vBid3'] + df['vBid4'] + df['vBid5']) / 5
df['average_asks_volume'] = (df['vAsk1'] + df['vAsk2'] + df['vAsk3'] + df['vAsk4'] + df['vAsk5']) / 5
df['difference'] = df['average_asks_volume'] - df['average_bids_volume']

# =============================================================================
# 3. Select Time Range
# =============================================================================
st.markdown("### 3. Select Time Range")
min_date = df.index.min().date()
max_date = df.index.max().date()

st.sidebar.header("Select Time Range (recommended: view one day at a time :date 14 - 16)")
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

if start_date > end_date:
    st.error("Start date must be before or equal to the end date.")
else:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_filtered = df.loc[start_ts:end_ts]
    if df_filtered.empty:
        st.warning("No data found in the selected date range.")
    else:
        # =============================================================================
        # 4. Create Background Shapes Based on Imbalance (difference)
        # =============================================================================
        st.markdown("### 4. Create Background Shapes Based on Imbalance")
        sign_segments = []
        prev_sign = None
        start_idx = None
        time_index = df_filtered.index.to_list()
        diff_values = df_filtered['difference'].to_list()
        for i in range(len(diff_values)):
            sign = (diff_values[i] >= 0)  # True = green, False = red
            if prev_sign is None:
                prev_sign = sign
                start_idx = i
            else:
                if sign != prev_sign:
                    sign_segments.append((time_index[start_idx], time_index[i], prev_sign))
                    prev_sign = sign
                    start_idx = i
        if start_idx is not None and prev_sign is not None:
            sign_segments.append((time_index[start_idx], time_index[-1], prev_sign))
        shapes = []
        for seg in sign_segments:
            x0, x1, s_val = seg
            fill_col = 'green' if s_val else 'red'
            shapes.append(
                dict(
                    type='rect',
                    xref='x',
                    x0=x0,
                    x1=x1,
                    yref='paper',
                    y0=0,
                    y1=1,
                    fillcolor=fill_col,
                    opacity=0.3,
                    line=dict(width=0)
                )
            )

        # =============================================================================
        # 5. Plot Price Chart (Bid1/Ask1) with Background Shapes
        # =============================================================================
        st.markdown("### 5. Price Chart: Bid1/Ask1 with Background Colors Indicating Imbalance")
        fig_bid_ask = go.Figure()
        fig_bid_ask.add_trace(
            go.Scatter(
                x=df_filtered.index,
                y=df_filtered['Bid1'],
                mode='lines',
                line=dict(color='blue'),
                name='Bid1'
            )
        )
        fig_bid_ask.add_trace(
            go.Scatter(
                x=df_filtered.index,
                y=df_filtered['Ask1'],
                mode='lines',
                line=dict(color='red'),
                name='Ask1'
            )
        )
        fig_bid_ask.update_layout(
            title="Bid1 / Ask1 with Full Background (Imbalance)",
            xaxis_title="Time",
            yaxis_title="Price",
            shapes=shapes,
            hovermode='x unified'
        )
        st.plotly_chart(fig_bid_ask, use_container_width=True)

        # =============================================================================
        # 6. Backtest Simulation: One Action per Row + SL = 5
        # =============================================================================
        st.markdown("### 6. Backtest Simulation (One Action per Row + Stop Loss = 5)")
        # Parameters
        contract_size = 200   # 200 baht/point for S50M24
        SL_points = 5         # Stop Loss = 5 points
        commission_fee = 40   # Commission fee per trade (open+close)
        lot_size = 1          # Lot size: 1 contract

        position = 0          # 0: no position, 1: Long, -1: Short
        entry_price = None
        entry_time = None
        trades = []
        last_trade_timestamp = None  # To ensure one action per unique timestamp

        for timestamp, row in df_filtered.iterrows():
            # Skip if this timestamp already had a trade action
            if last_trade_timestamp == timestamp:
                continue

            current_signal = (row['difference'] >= 0)  # True: signal green, False: signal red

            # If a position is open, check exit conditions
            if position == 1:  # Long
                # Check Stop Loss for Long: if Bid1 <= entry_price - SL_points
                if row['Bid1'] <= entry_price - SL_points:
                    exit_price = row['Bid1']
                    exit_time = timestamp
                    profit = (exit_price - entry_price) * contract_size * lot_size
                    profit_net = profit - commission_fee
                    trades.append({
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Position': 'Long',
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Profit': profit_net,
                        'Remark': 'SL hit'
                    })
                    position = 0
                    last_trade_timestamp = timestamp
                    continue
                # Check Reversal for Long: if signal changes to red
                if not current_signal:
                    exit_price = row['Bid1']
                    exit_time = timestamp
                    profit = (exit_price - entry_price) * contract_size * lot_size
                    profit_net = profit - commission_fee
                    trades.append({
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Position': 'Long',
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Profit': profit_net,
                        'Remark': 'Reversal'
                    })
                    position = 0
                    last_trade_timestamp = timestamp
                    continue

            elif position == -1:  # Short
                # Check Stop Loss for Short: if Ask1 >= entry_price + SL_points
                if row['Ask1'] >= entry_price + SL_points:
                    exit_price = row['Ask1']
                    exit_time = timestamp
                    profit = (entry_price - exit_price) * contract_size * lot_size
                    profit_net = profit - commission_fee
                    trades.append({
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Position': 'Short',
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Profit': profit_net,
                        'Remark': 'SL hit'
                    })
                    position = 0
                    last_trade_timestamp = timestamp
                    continue
                # Check Reversal for Short: if signal changes to green
                if current_signal:
                    exit_price = row['Ask1']
                    exit_time = timestamp
                    profit = (entry_price - exit_price) * contract_size * lot_size
                    profit_net = profit - commission_fee
                    trades.append({
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Position': 'Short',
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Profit': profit_net,
                        'Remark': 'Reversal'
                    })
                    position = 0
                    last_trade_timestamp = timestamp
                    continue

            # If no position is open, check for entry conditions
            if position == 0:
                if current_signal:
                    # Open Long at Ask1
                    position = 1
                    entry_price = row['Ask1']
                    entry_time = timestamp
                    last_trade_timestamp = timestamp
                else:
                    # Open Short at Bid1
                    position = -1
                    entry_price = row['Bid1']
                    entry_time = timestamp
                    last_trade_timestamp = timestamp

        # Close any open position at the end of the data
        if position != 0:
            last_row = df_filtered.iloc[-1]
            exit_time = df_filtered.index[-1]
            if position == 1:
                exit_price = last_row['Bid1']
                profit = (exit_price - entry_price) * contract_size * lot_size
                profit_net = profit - commission_fee
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Position': 'Long',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Profit': profit_net,
                    'Remark': 'End'
                })
            else:
                exit_price = last_row['Ask1']
                profit = (entry_price - exit_price) * contract_size * lot_size
                profit_net = profit - commission_fee
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Position': 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Profit': profit_net,
                    'Remark': 'End'
                })
            position = 0
            last_trade_timestamp = exit_time

        # Create Trade Log DataFrame
        trades_df = pd.DataFrame(trades)
        # Convert Entry/Exit Times to string format with microsecond precision
        trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        total_profit = trades_df['Profit'].sum() if not trades_df.empty else 0

        st.subheader("Trade Log")
        st.dataframe(trades_df)
        st.write(f"**Total Net Profit:** {total_profit:,.2f} Baht")

        # =============================================================================
        # 7. Plot Entry/Exit Points on Price Chart with Distinct Markers for Long/Short
        # =============================================================================
        if not trades_df.empty:
            # Separate trades into Long and Short groups
            long_trades = trades_df[trades_df['Position'].str.contains('Long', case=False)]
            short_trades = trades_df[trades_df['Position'].str.contains('Short', case=False)]

            # Extract entry/exit times and prices for Long trades
            long_entry_times = long_trades['Entry Time']
            long_entry_prices = long_trades['Entry Price']
            long_exit_times = long_trades['Exit Time']
            long_exit_prices = long_trades['Exit Price']

            # Extract entry/exit times and prices for Short trades
            short_entry_times = short_trades['Entry Time']
            short_entry_prices = short_trades['Entry Price']
            short_exit_times = short_trades['Exit Time']
            short_exit_prices = short_trades['Exit Price']

            # Long Entry: Green Circle
            fig_bid_ask.add_trace(
                go.Scatter(
                    x=long_entry_times,
                    y=long_entry_prices,
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='green'),
                    name='Long Entry'
                )
            )
            # Long Exit: Green Cross
            fig_bid_ask.add_trace(
                go.Scatter(
                    x=long_exit_times,
                    y=long_exit_prices,
                    mode='markers',
                    marker=dict(symbol='x', size=12, color='green'),
                    name='Long Exit'
                )
            )
            # Short Entry: Dark Red Circle
            fig_bid_ask.add_trace(
                go.Scatter(
                    x=short_entry_times,
                    y=short_entry_prices,
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='darkred'),
                    name='Short Entry'
                )
            )
            # Short Exit: Dark Red Cross
            fig_bid_ask.add_trace(
                go.Scatter(
                    x=short_exit_times,
                    y=short_exit_prices,
                    mode='markers',
                    marker=dict(symbol='x', size=12, color='darkred'),
                    name='Short Exit'
                )
            )

            fig_bid_ask.update_layout(
                title="Bid1 / Ask1 with BG + Colored Entry/Exit Points",
                hovermode='x unified'
            )
            st.plotly_chart(fig_bid_ask, use_container_width=True)
