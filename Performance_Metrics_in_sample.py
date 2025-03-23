import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title("Performance Metrics of Strategy")

# -------------------------------------------------------------------------
# 1) โหลดข้อมูล
# -------------------------------------------------------------------------
df = pd.read_csv('in_sample.csv')

# รวมคอลัมน์ Date และ Time_ เป็น Datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time_'])
df.sort_values('Datetime', inplace=True)
df.set_index('Datetime', inplace=True)

# (หากต้องการ Resample ให้ใช้ df = df.resample('1Min').agg({...}) ที่นี่ก่อน)

# -------------------------------------------------------------------------
# 2) คำนวณ Average Bids/Asks Volume และ Imbalance (difference)
# -------------------------------------------------------------------------
df['average_bids_volume'] = (df['vBid1'] + df['vBid2'] + df['vBid3'] + df['vBid4'] + df['vBid5']) / 5
df['average_asks_volume'] = (df['vAsk1'] + df['vAsk2'] + df['vAsk3'] + df['vAsk4'] + df['vAsk5']) / 5

df['difference'] = df['average_asks_volume'] - df['average_bids_volume']  # >= 0 = green, < 0 = red

# -------------------------------------------------------------------------
# 3) เลือกช่วงเวลา (Time Range) ผ่าน Sidebar
# -------------------------------------------------------------------------
min_date = df.index.min().date()
max_date = df.index.max().date()

st.sidebar.header("Select Time Range (recommended: view one day at a time :date 2 - 14")
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

if start_date > end_date:
    st.error("วันเริ่มต้นต้องไม่เกินวันสิ้นสุด")
else:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    df_filtered = df.loc[start_ts:end_ts]

    if df_filtered.empty:
        st.warning("ไม่มีข้อมูลในช่วงวันที่เลือก")
    else:
        # ---------------------------------------------------------------------
        # 4) สร้าง Shapes สำหรับพื้นหลังสี (เขียว/แดง) ตามค่า difference
        # ---------------------------------------------------------------------
        sign_segments = []
        prev_sign = None
        start_idx = None

        time_index = df_filtered.index.to_list()
        diff_values = df_filtered['difference'].to_list()

        for i in range(len(diff_values)):
            sign = (diff_values[i] >= 0)  # True=green, False=red
            if prev_sign is None:
                prev_sign = sign
                start_idx = i
            else:
                if sign != prev_sign:
                    sign_segments.append((time_index[start_idx], time_index[i], prev_sign))
                    prev_sign = sign
                    start_idx = i
        # ปิด segment สุดท้าย
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

        # ---------------------------------------------------------------------
        # 5) Plot กราฟ Bid1 / Ask1 พร้อมพื้นหลังสี
        # ---------------------------------------------------------------------
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
            title="Bid1 / Ask1 with Full BG (Imbalance)",
            xaxis_title="Time",
            yaxis_title="Price",
            shapes=shapes,
            hovermode='x unified'
        )

        st.plotly_chart(fig_bid_ask, use_container_width=True)

        # ---------------------------------------------------------------------
        # 6) Backtest: One Action per Row + Stop Loss = 5
        # ---------------------------------------------------------------------
        st.header("Backtest Simulation (One Action per Row + SL = 5)")

        # พารามิเตอร์
        contract_size = 200   # S50M24 = 200 บาท/จุด
        SL_points = 5         # SL = 5 จุด
        commission_fee = 40   # บาท/ต่อการเทรด (เปิด+ปิด)
        lot_size = 1          # สมมติคำนวณได้ 1 สัญญา

        position = 0          # 0=ไม่มีสถานะ, 1=Long, -1=Short
        entry_price = None
        entry_time = None

        trades = []
        last_trade_timestamp = None  # เก็บ timestamp ที่เพิ่งทำการเทรด (ปิด/เปิด)

        # วนลูปแบบ One Action per Row
        for timestamp, row in df_filtered.iterrows():
            # หาก timestamp นี้ == trade ล่าสุด -> ข้าม
            if last_trade_timestamp == timestamp:
                continue

            current_signal = (row['difference'] >= 0)  # True=green, False=red

            # ถ้ามีสถานะ -> ตรวจเฉพาะ "ปิด"
            if position == 1:  # Long
                # Stop Loss
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

                # Reversal
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
                # Stop Loss
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

                # Reversal
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

            # ถ้าไม่มีสถานะ -> ตรวจเฉพาะ "เปิด"
            if position == 0:
                if current_signal:
                    # เปิด Long
                    position = 1
                    entry_price = row['Ask1']
                    entry_time = timestamp
                    last_trade_timestamp = timestamp
                else:
                    # เปิด Short
                    position = -1
                    entry_price = row['Bid1']
                    entry_time = timestamp
                    last_trade_timestamp = timestamp

        # ปิดสถานะค้างเมื่อถึงท้ายข้อมูล
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
            else:  # Short
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

        # สร้าง DataFrame ของการเทรด
        trades_df = pd.DataFrame(trades)
        total_profit = trades_df['Profit'].sum() if not trades_df.empty else 0

        trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        st.subheader("Trade Log")
        st.dataframe(trades_df)


# เรียงข้อมูลตาม Exit Time (หรือ Entry Time)
trades_df = trades_df.sort_values('Exit Time')

# -----------------------------------------------------------------------------------
# 1. Profit Factor: สัดส่วนของกำไรรวมต่อขาดทุนรวม (เฉพาะ trade ที่ได้กำไรและขาดทุน)
# -----------------------------------------------------------------------------------
total_profit = trades_df.loc[trades_df['Profit'] > 0, 'Profit'].sum()
total_loss = trades_df.loc[trades_df['Profit'] < 0, 'Profit'].sum()
profit_factor = total_profit / abs(total_loss) if total_loss != 0 else np.nan

# -----------------------------------------------------------------------------------
# 2. Win Rate: อัตราการชนะของเทรดทั้งหมด
# -----------------------------------------------------------------------------------
win_rate = len(trades_df[trades_df['Profit'] > 0]) / len(trades_df) if len(trades_df) > 0 else np.nan

# -----------------------------------------------------------------------------------
# 3. Expectancy: ผลตอบแทนเฉลี่ยต่อเทรด
# -----------------------------------------------------------------------------------
expectancy = trades_df['Profit'].mean()

# -----------------------------------------------------------------------------------
# 4. สร้าง Equity Curve จาก Trade Log
# -----------------------------------------------------------------------------------
initial_capital = 100000  # บาท
trades_df['Cumulative Equity'] = initial_capital + trades_df['Profit'].cumsum()

# -----------------------------------------------------------------------------------
# 5. Sharpe Ratio (Trade-Based)
# -----------------------------------------------------------------------------------
# คำนวณผลตอบแทน (Return) ระหว่างแต่ละเทรดจาก Equity Curve
returns = trades_df['Cumulative Equity'].pct_change().dropna()
# Sharpe Ratio = (mean return / std return) * sqrt(n)
sharpe_ratio = np.sqrt(len(returns)) * returns.mean() / returns.std() if returns.std() != 0 else np.nan

# -----------------------------------------------------------------------------------
# 6. Maximum Drawdown (MDD)
# -----------------------------------------------------------------------------------
rolling_max = trades_df['Cumulative Equity'].cummax()
drawdown = (trades_df['Cumulative Equity'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# -----------------------------------------------------------------------------------
# แสดงผลตัวชี้วัดด้วย Streamlit
# -----------------------------------------------------------------------------------
st.markdown("""
## Displaying Performance Metrics with Streamlit

After the backtest simulation is complete, several key performance metrics are calculated to evaluate the trading strategy. These metrics include:

- **Profit Factor:**  
  The ratio of the total profits (sum of all winning trades) to the absolute value of total losses (sum of all losing trades).  
  A higher profit factor indicates that the strategy generates more profit relative to its losses.

- **Win Rate:**  
  The percentage of trades that are profitable.  
  This metric helps assess how consistently the strategy is successful.

- **Expectancy (Average Profit per Trade):**  
  The average profit per trade, which provides insight into the average return generated from each trade.

- **Sharpe Ratio:**  
  A risk-adjusted return measure calculated from the percentage changes in the cumulative equity (equity curve).  
  It is computed as:  
  \[
  \text{Sharpe Ratio} = \frac{\text{mean return}}{\text{std return}} \times \sqrt{n}
  \]
  where \( n \) is the number of returns. A higher Sharpe ratio indicates better risk-adjusted performance.

- **Maximum Drawdown (MDD):**  
  The maximum decline from a peak to a trough in the equity curve, expressed as a percentage.  
  It provides an estimate of the worst-case loss scenario during the backtesting period.

Additionally, the **Equity Curve** is plotted to visualize the cumulative performance of the strategy over time. This helps in understanding how the capital evolves and in identifying periods of significant drawdown.
""")

st.subheader("Performance Metrics")
st.write("**Profit Factor:**", profit_factor)
st.write("**Win Rate:**", win_rate)
st.write("**Expectancy (Average Profit per Trade):**", expectancy)
st.write("**Sharpe Ratio:**", sharpe_ratio)
st.write("**Max Drawdown:**", max_drawdown)

# แสดง Equity Curve
st.subheader("Equity Curve")
st.line_chart(trades_df.set_index('Exit Time')['Cumulative Equity'])
