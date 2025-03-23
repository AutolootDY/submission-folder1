import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit layout
st.set_page_config(page_title="Financial Data Analysis", layout="wide")
st.title("Financial Data Analysis with Streamlit")

# --------------------------------------------------------------------------------
# 1. Read data from CSV
# --------------------------------------------------------------------------------
st.markdown("""
### 1. Read Data from CSV  
In this section, we load the data from **in_sample.csv** located in the same directory as the script.  
The file includes columns such as **Date**, **Time_**, **Bid1**, **Ask1**, **vBid1**, **vAsk1**, etc.
""")
csv_file = 'in_sample.csv'
df = pd.read_csv(csv_file)

# --------------------------------------------------------------------------------
# 2. Process Date and Set Index
# --------------------------------------------------------------------------------
st.markdown("""
### 2. Process Date and Set Index  
- Combine the **Date** and **Time_** columns to create a **Datetime** column  
- Convert the **Datetime** column to datetime objects using `pd.to_datetime`  
- Set **Datetime** as the DataFrame index and sort the data by time to support time series analysis
""")
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time_'])
df.set_index('Datetime', inplace=True)
df.sort_index(inplace=True)

st.subheader("Sample Data (Table Head):")
st.dataframe(df.head())

# --------------------------------------------------------------------------------
# 3. Time Range Selection via Sidebar
# --------------------------------------------------------------------------------
min_date = df.index.min().date()
max_date = df.index.max().date()

st.sidebar.header("Select Time Range (recommended: view one day at a time :date 2 - 14")
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

if start_date > end_date:
    st.error("Start Date must not exceed End Date")
else:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_filtered = df.loc[start_ts:end_ts]
    
    if df_filtered.empty:
        st.warning("No data in the selected date range")
    else:
        st.write(f"Displaying data from {start_date} to {end_date}")
        
        # --------------------------------------------------------------------------------
        # 4. Display Basic Statistics for the Selected Date Range
        # --------------------------------------------------------------------------------
        st.markdown("""
        ### 4. Basic Statistics for the Selected Date Range  
        Use `df.describe()` to display statistics such as mean, max, min, and standard deviation.
        """)
        st.dataframe(df_filtered.describe())
        
        # --------------------------------------------------------------------------------
        # 5. Create New Variables: MidPrice and Spread
        # --------------------------------------------------------------------------------
        st.markdown("""
        ### 5. Create New Variables: **MidPrice** and **Spread**  
        - **MidPrice:** Calculated as the average of **Bid1** and **Ask1**, representing the market's mid-price.  
        - **Spread:** Calculated as the difference between **Ask1** and **Bid1**, measuring the price gap.
        """)
        df_filtered['MidPrice'] = (df_filtered['Bid1'] + df_filtered['Ask1']) / 2
        df_filtered['Spread'] = df_filtered['Ask1'] - df_filtered['Bid1']
        
        # --------------------------------------------------------------------------------
        # 6. Visualization: Price Changes (Bid1, Ask1, MidPrice)
        # --------------------------------------------------------------------------------
        st.markdown("""
        ### 6. Visualization: Price Changes  
        The graph below shows the changes in **Bid1**, **Ask1**, and **MidPrice** over time.
        """)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df_filtered.index, df_filtered['Bid1'], label='Bid1', alpha=0.7)
        ax.plot(df_filtered.index, df_filtered['Ask1'], label='Ask1', alpha=0.7)
        ax.plot(df_filtered.index, df_filtered['MidPrice'], label='MidPrice', linestyle='--', color='black')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # --------------------------------------------------------------------------------
        # 7. Visualization: Distribution of Spread
        # --------------------------------------------------------------------------------
        st.markdown("""
        ### 7. Visualization: Distribution of **Spread**  
        The histogram with KDE below displays the distribution of **Spread**.
        """)
        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.histplot(df_filtered['Spread'], bins=50, kde=True, color='orange', ax=ax2)
        ax2.set_xlabel('Spread')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)
        
        # --------------------------------------------------------------------------------
        # 8. Visualization: Volume of Bid1 and Ask1 over Time
        # --------------------------------------------------------------------------------
        st.markdown("""
        ### 8. Visualization: Volume of **vBid1** and **vAsk1** over Time  
        This graph shows the changes in volume for **vBid1** and **vAsk1** over time.
        """)
        fig3, ax3 = plt.subplots(figsize=(12,6))
        ax3.plot(df_filtered.index, df_filtered['vBid1'], label='vBid1')
        ax3.plot(df_filtered.index, df_filtered['vAsk1'], label='vAsk1')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Volume')
        ax3.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig3)
        
        # --------------------------------------------------------------------------------
        # 9. Visualization: Correlation Matrix among Key Variables
        # --------------------------------------------------------------------------------
        st.markdown("""
        ### 9. Visualization: Correlation Matrix among Key Variables  
        The heatmap below shows the correlation among key variables such as **Bid1**, **Ask1**, **vBid1**, **vAsk1**, **MidPrice**, and **Spread**.
        """)
        cols_of_interest = ['Bid1', 'Ask1', 'vBid1', 'vAsk1', 'MidPrice', 'Spread']
        corr = df_filtered[cols_of_interest].corr()
        fig4, ax4 = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)
        
        # --------------------------------------------------------------------------------
        # 10. Create New Variables for Imbalance Analysis
        # --------------------------------------------------------------------------------
        st.markdown("""
        ### 10. Create New Variables for Imbalance Analysis  
        - **Average Bids Volume:** The average of **vBid1** to **vBid5**  
        - **Average Asks Volume:** The average of **vAsk1** to **vAsk5**  
        - **Difference (Imbalance):** The difference between **Average Asks Volume** and **Average Bids Volume**
        """)
        df_filtered['average_bids_volume'] = (df_filtered['vBid1'] + df_filtered['vBid2'] + df_filtered['vBid3'] + df_filtered['vBid4'] + df_filtered['vBid5']) / 5
        df_filtered['average_asks_volume'] = (df_filtered['vAsk1'] + df_filtered['vAsk2'] + df_filtered['vAsk3'] + df_filtered['vAsk4'] + df_filtered['vAsk5']) / 5
        df_filtered['difference'] = df_filtered['average_asks_volume'] - df_filtered['average_bids_volume']
        
        # --------------------------------------------------------------------------------
        # 11. Prepare Background Color Segments Based on the Sign of Difference
        # --------------------------------------------------------------------------------
        time_index = df_filtered.index.to_list()
        diff_values = df_filtered['difference'].tolist()

        sign_segments = []
        prev_sign = None
        start_idx = None

        for i in range(len(diff_values)):
            current_sign = diff_values[i] >= 0  # True if positive or zero, False if negative
            if prev_sign is None:
                prev_sign = current_sign
                start_idx = i
            else:
                if current_sign != prev_sign:
                    sign_segments.append((time_index[start_idx], time_index[i], prev_sign))
                    prev_sign = current_sign
                    start_idx = i
        if start_idx is not None and prev_sign is not None:
            sign_segments.append((time_index[start_idx], time_index[-1], prev_sign))
        
        # --------------------------------------------------------------------------------
        # 12. Visualization: Graphs with Background Color Segments
        #     - Graph 1: Average Bids Volume vs Average Asks Volume (No Background Color)
        #     - Graph 2: Imbalance (Difference) with Full BG Color
        #     - Graph 3: Bid1 / Ask1 with Full BG Color (Imbalance)
        # --------------------------------------------------------------------------------
        st.markdown("""
        ### 12. Visualization: Graphs with Background Color Segments  
        Below are 3 graphs:
        1. **Graph 1:** Average Bids Volume vs Average Asks Volume (No Background Color)  
        2. **Graph 2:** Imbalance (Difference) with Full BG Color  
        3. **Graph 3:** Bid1 / Ask1 with Full BG Color (Imbalance)
        """)
        
        # Graph 1: Average Bids Volume vs Average Asks Volume (No Background Color)
        fig5, ax5 = plt.subplots(figsize=(12,6))
        ax5.plot(df_filtered.index, df_filtered['average_bids_volume'], label='Average Bids Volume', color='blue')
        ax5.plot(df_filtered.index, df_filtered['average_asks_volume'], label='Average Asks Volume', color='orange')
        ax5.set_title("Graph 1: Average Bids Volume vs Average Asks Volume")
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Volume")
        ax5.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig5)
        st.markdown("""
        ### Graph 1: Average Bids Volume vs Average Asks Volume
        - **Description:**  
          This graph displays the average volume of bids and asks, calculated from 5 levels (vBid1 to vBid5 and vAsk1 to vAsk5) over time.  
          Calculation: It sums the volumes at each level and divides by 5 to obtain the average.
        - **Benefit:**  
          Provides an overview of trading volumes on both the bid and ask sides, indicating whether the market is balanced or diverging—essential information for trading decisions.
        """)
        
        # Graph 2: Imbalance (Difference) with Full BG Color
        fig6, ax6 = plt.subplots(figsize=(12,6))
        for seg in sign_segments:
            x0, x1, sign_val = seg
            color = 'green' if sign_val else 'red'
            ax6.axvspan(x0, x1, facecolor=color, alpha=0.3)
        ax6.plot(df_filtered.index, df_filtered['difference'], label='Difference (Imbalance)', color='blue')
        ax6.set_title("Graph 2: Imbalance (Difference) with Full BG Color")
        ax6.set_xlabel("Time")
        ax6.set_ylabel("Difference")
        ax6.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig6)
        st.markdown("""
        ### Graph 2: Imbalance (Difference) with Full BG Color
        - **Description:**  
          This graph shows the imbalance (difference) between the average asks volume and average bids volume over time, with a background color that reflects the sign (green for positive/zero and red for negative).  
          Calculation: For example, difference = average_asks_volume - average_bids_volume.
        - **Benefit:**  
          Clearly indicates which side the market is leaning towards during different periods, providing signals for trading decisions and market trend analysis.
        """)
        
        # Graph 3: Bid1 / Ask1 with Full BG Color (Imbalance)
        fig7, ax7 = plt.subplots(figsize=(12,6))
        for seg in sign_segments:
            x0, x1, sign_val = seg
            color = 'green' if sign_val else 'red'
            ax7.axvspan(x0, x1, facecolor=color, alpha=0.3)
        ax7.plot(df_filtered.index, df_filtered['Bid1'], label='Bid1', color='blue')
        ax7.plot(df_filtered.index, df_filtered['Ask1'], label='Ask1', color='red')
        ax7.set_title("Graph 3: Bid1 / Ask1 with Full BG Color (Imbalance)")
        ax7.set_xlabel("Time")
        ax7.set_ylabel("Price")
        ax7.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig7)
        st.markdown("""
        ### Graph 3: Bid1 / Ask1 with Full BG Color (Imbalance)
        - **Description:**  
          This graph displays the best bid (Bid1) and ask (Ask1) prices over time, with a background color indicating market imbalance (the same segments as in Graph 2).
        - **Benefit:**  
          It helps users observe primary price movements along with the imbalance signal, which is crucial for analyzing market trends and timing trades.
        """)
        
        st.success("Preliminary analysis is complete!")
        st.markdown("""
        The results from this analysis can serve as a basis for developing trading strategies, such as detecting order imbalances, assessing spread volatility, and monitoring volume trends.
        """)
