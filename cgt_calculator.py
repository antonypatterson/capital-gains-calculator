# %%
import pandas as pd
import numpy as np
#from forex_python.converter import CurrencyRates
from datetime import datetime

# %%
# Function to get the date range from the tax year
def get_date_range_from_year(tax_year: int):
    start_date = datetime(tax_year - 1, 7, 1)  # July 1st of the previous year
    end_date = datetime(tax_year, 6, 30)        # June 30th of the current year
    return start_date, end_date

# Function to calculate capital gains using FIFO
def calculate_capital_gains(df_all_trades, ticker_code, tax_year: int = None):
    # Filter for transactions of the given ticker code
    df_ticker = df_all_trades[df_all_trades['Ticker code'] == ticker_code].copy()

    # Fill NA values
    df_ticker['Quantity'] = df_ticker['Quantity'].fillna(0)
    df_ticker['Price_AUD'] = df_ticker['Price_AUD'].fillna(0)
    
    # Split the data into buys and sells
    buys = df_ticker[df_ticker["Transaction type"] == "BUY"]
    sells = df_ticker[df_ticker["Transaction type"] == "SELL"]
    
    # Convert Trade date to datetime
    df_ticker['Trade date'] = pd.to_datetime(df_ticker['Trade date'], format="%d/%m/%Y")

    # List to track the remaining FIFO quantities from buys
    buy_queue = []
    
    # Variables to track total capital gains
    total_capital_gains_without_discount = 0
    total_capital_gains_with_discount = 0
    
    # Process each buy to populate the buy queue with purchase date
    for _, buy in buys.iterrows():
        buy_queue.append({
            "Quantity": buy["Quantity"],
            "Price_AUD": buy["Price_AUD"],
            "Date": buy["Trade date"],  # Store the buy date directly
            "Transaction fee_AUD": buy['Transaction fee_AUD']
        })
    
    # Get the date range if a tax year is specified
    if tax_year is not None:
        start_date, end_date = get_date_range_from_year(tax_year)

    # Process each sell
    for _, sell in sells.iterrows():
        sell_qty = sell["Quantity"]
        sell_price = sell["Price_AUD"]
        sell_date = sell["Trade date"]
        sell_trans_fee = sell['Transaction fee_AUD']
        
        remaining_sell_qty = sell_qty

        cumul_gains_per_row_discount = 0
        cumul_gains_per_row_non_discount = 0
        # Process the FIFO logic with the buy queue
        while remaining_sell_qty > 0 and buy_queue:
            buy = buy_queue.pop(0)
            buy_qty = buy["Quantity"]
            buy_price = buy["Price_AUD"]
            buy_date = buy["Date"]
            buy_fee = buy['Transaction fee_AUD']
            
            # Determine the quantity to sell from this buy
            if buy_qty <= remaining_sell_qty:
                quantity_sold = buy_qty
                remaining_sell_qty -= buy_qty
            else:
                quantity_sold = remaining_sell_qty
                buy["Quantity"] = buy_qty - remaining_sell_qty
                buy_qty_utilised = buy["Quantity"]
                remaining_sell_qty = 0
                buy_queue.insert(0, buy)  # Reinsert modified buy

            # Calculate capital gains without discount
            #capital_gain_without_discount = quantity_sold * (sell_price - buy_price)   
            sell_proceeds_per_unit = sell_price - (sell_trans_fee / sell_qty)
            buy_cost_per_unit = buy_price + (buy_fee / buy_qty)
            capital_gain_without_discount = quantity_sold * (sell_proceeds_per_unit - buy_cost_per_unit)

            # Also subtract the cost base from the capital gains   
            #capital_gain_without_discount -= (buy_fee / buy_qty)

            # Calculate capital gains with discount based on holding period
            holding_period = (sell_date - buy_date).days
            if (holding_period > 365) and (capital_gain_without_discount > 0):  # Held for over one year
                capital_gain_with_discount = 0.5 * capital_gain_without_discount
            else:
                capital_gain_with_discount = capital_gain_without_discount

            cumul_gains_per_row_discount += capital_gain_with_discount
            cumul_gains_per_row_non_discount += capital_gain_without_discount
            
        # Check if the sell is within the tax year range
        #print(f"start date is {start_date}, sell date is {sell_date}, end date is {end_date}")
        #print(f"boolean is {start_date <= sell_date <= end_date}")
        if tax_year is not None:
            if start_date <= sell_date <= end_date:
                total_capital_gains_with_discount += cumul_gains_per_row_discount
                total_capital_gains_without_discount += cumul_gains_per_row_non_discount
        else: # i.e tax year is None
                total_capital_gains_with_discount += cumul_gains_per_row_discount
                total_capital_gains_without_discount += cumul_gains_per_row_non_discount


    # If there's leftover sell quantity, warn the user
    #if remaining_sell_qty > 0:
        #print(f"Warning: Not enough buys to match sell of {remaining_sell_qty} units for {ticker_code}")

    return (total_capital_gains_without_discount, total_capital_gains_with_discount)

# %%
# Vectorized conversion function to add new 'AUD Amount' column
def convert_to_aud_vectorized(row, converter):
    if row["Currency"].lower() == "usd":
        # Get the historical AUD/USD rate for the transaction date
        rate = converter.get_rate('USD', 'AUD', row["Trade date"])
        return row["Amount"] * rate  # Convert USD to AUD
    else:
        return row["Amount"]  # No conversion needed if already in AUD

# %%
def convert_to_aud_from_xlsx(row, date):
    pass

# %%
# Read in exchange rates
exchange_1 = pd.read_csv('exchange-18-to-22.csv', parse_dates=['Series ID'])
exchange_2 = pd.read_csv('exchange-23-to-24.csv', parse_dates=['Series ID'])
exchange_concat = pd.concat([exchange_1, exchange_2]).loc[:, ['Series ID', 'FXRUSD']].rename(columns={'Series ID': 'Exchange date'})
exchange_concat['USD_to_AUD'] = 1 / exchange_concat['FXRUSD']
exchange_concat = exchange_concat.drop(columns={'FXRUSD'})
exchange_concat.tail()

# Default frequency includes all days
date_exchanges = pd.date_range(start='2018-01-01', end=pd.Timestamp.today())  
date_df = pd.DataFrame(date_exchanges, columns=['Exchange date'])
merged_df = pd.merge(date_df, exchange_concat, how='left', on='Exchange date')


# now use linear interpolation to fill in missing values
merged_df['USD_to_AUD'] = merged_df['USD_to_AUD'].interpolate(method='linear')
merged_df.head()
merged_df.to_csv('outputs/exchange rates_updated.csv')
exchange_concat = merged_df.copy()

# %%
# Load the CSV data
df = pd.read_csv("shares_raw.csv", parse_dates=["Trade date"])


# %%
# Ensure the dataframe is sorted by Trade date
df = df.sort_values("Trade date")
df = pd.merge(df, exchange_concat, how='left', left_on="Trade date", right_on="Exchange date")


# %%
na_rows_usd = df[df['USD_to_AUD'].isna() & (df['Currency'].str.lower == 'usd')]
na_rows = df[df['USD_to_AUD'].isna()].shape[0]
df['Amount incl transaction'] = np.where(df['Transaction type'] == 'BUY', df['Amount'] + df['Transaction fee'], df['Amount'] - df['Transaction fee'])
df['Amount_AUD'] = np.where(df['Currency'].str.lower() == "usd", df['Amount'] * df['USD_to_AUD'], df['Amount'])
df['Amount_AUD incl transaction'] = np.where(df['Currency'].str.lower() == "usd", df['Amount incl transaction'] * df['USD_to_AUD'], df['Amount incl transaction'])
df['Transaction fee_AUD'] = np.where(df['Currency'].str.lower() == "usd", df['Transaction fee'] * df['USD_to_AUD'], df['Transaction fee'])
df['Price_AUD'] = np.where(df['Currency'].str.lower() == "usd", df['Price'] * df['USD_to_AUD'], df['Price'])
df['Ticker code'] = df['Market code'] + ":" + df['Instrument code']
df.head()

# %%
df.to_csv('outputs/shares_out.csv')

# %%
# Create subset ticker listing for all tickers with at least 1 SELL entry
sell_trades = df.loc[df['Transaction type'] == 'SELL', ['Ticker code', 'Transaction type']]
ticker_codes_sold = sell_trades['Ticker code'].unique().tolist()


# %%
sell_trades.head()

# %%
# Calculate capital gains for each ticker code
results = []
for ticker in ticker_codes_sold:
    count_ticker = df[df['Ticker code'] == ticker].shape[0]
    capital_gains = calculate_capital_gains(df, ticker, tax_year=2024)
    results.append({
        'Ticker code': ticker,
        'Capital Gains without Discount': capital_gains[0],
        'Capital Gains with Discount': capital_gains[1]
    })

# Create a DataFrame from the results
capital_gains_df = pd.DataFrame(results)

# Display the DataFrame
total_cap_gains_no_disc = capital_gains_df['Capital Gains without Discount'].sum()
print(capital_gains_df)
print(f"Total gains without discount applied: {total_cap_gains_no_disc}")

# Export results to .csv file
capital_gains_df.to_csv('outputs/capital_gains_FY24.csv')


# %% [markdown]
# ##### currency rates package doesn't work so disregard below until that's fixed. otherwise just use excel extract

# %%
# initialise forex converter and insert new row for AUD amounts
#c = CurrencyRates()
#df.columns = df.columns.str.strip()
#myDate = df.loc[2, 'Trade date']
#print(myDate)
#cverted = c.get_rates('USD')
#print(converted)
#df['AUD Amount'] = 5#df.apply(lambda x: convert_to_aud_vectorized(x, c), axis=1)
#df.head()

# %%



