import pandas as pd

# --- Step 1 & 2: Initial EDA ---
# This part loads the data and gives us our starting point.
file_name = 'online_retail.csv'

try:
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    print("✅ File loaded successfully!")
    print("\n----------- Initial Data Info (Before Cleaning) -----------")
    df.info()
    print("\n----------- Initial Statistical Summary (Before Cleaning) -----------")
    print(df.describe())

except FileNotFoundError:
    print(f"❌ Error: The file '{file_name}' was not found.")
    exit()


# --- Step 3: Data Cleaning ---
# This new section cleans the data based on our findings from the initial EDA.
print("\n----------- Cleaning Data -----------")

# Print the shape of the dataframe before cleaning
print(f"Original shape: {df.shape}")

# 1. Drop rows with missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)
print(f"Shape after dropping null CustomerIDs: {df.shape}")

# 2. Remove canceled orders (Quantity > 0)
df = df[df['Quantity'] > 0]
print(f"Shape after removing negative quantities: {df.shape}")

# 3. Remove items with a price of 0
df = df[df['UnitPrice'] > 0]
print(f"Shape after removing zero unit prices: {df.shape}")

# 4. Convert CustomerID to integer for consistency
df['CustomerID'] = df['CustomerID'].astype(int)

# 5. Convert InvoiceDate to datetime objects
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("\n----------- Final Cleaned Data Info -----------")
# Display info to confirm all our changes
df.info()

print("\n----------- Final Cleaned Statistical Summary -----------")
# Display statistical summary to confirm changes (e.g., min Quantity is now 1)
print(df.describe())
# --- Step 4: Feature Engineering for Time Series ---
print("\n----------- Engineering Features for Forecasting -----------")

# Create a 'TotalSales' column
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# We need to aggregate sales data by day.
# We will set the InvoiceDate as the index to perform time-based grouping.
daily_sales = df.set_index('InvoiceDate').resample('D')['TotalSales'].sum().reset_index()

print("\n----------- Daily Sales Data -----------")
print(daily_sales.head())

print(f"\nWe now have {daily_sales.shape[0]} days of sales data to analyze.")
# --- Step 4: Feature Engineering for Time Series ---
print("\n----------- Engineering Features for Forecasting -----------")

# Create a 'TotalSales' column
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# We need to aggregate sales data by day.
# We will set the InvoiceDate as the index to perform time-based grouping.
daily_sales = df.set_index('InvoiceDate').resample('D')['TotalSales'].sum().reset_index()

print("\n----------- Daily Sales Data -----------")
print(daily_sales.head())

print(f"\nWe now have {daily_sales.shape[0]} days of sales data to analyze.")
# --- Step 4: Feature Engineering for Time Series ---
print("\n----------- Engineering Features for Forecasting -----------")

# Create a 'TotalSales' column
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# We need to aggregate sales data by day.
# We will set the InvoiceDate as the index to perform time-based grouping.
daily_sales = df.set_index('InvoiceDate').resample('D')['TotalSales'].sum().reset_index()

print("\n----------- Daily Sales Data -----------")
print(daily_sales.head())

print(f"\nWe now have {daily_sales.shape[0]} days of sales data to analyze.")