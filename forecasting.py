import pandas as pd
from prophet import Prophet

# --- Step 1: Load and Prepare Data ---
# We start with the same data loading and cleaning as before.
file_name = 'online_retail.csv'
try:
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    
    # --- Data Cleaning ---
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # --- Feature Engineering ---
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    daily_sales = df.set_index('InvoiceDate').resample('D')['TotalSales'].sum().reset_index()

    # Prophet requires columns to be named 'ds' (date) and 'y' (value)
    daily_sales.rename(columns={'InvoiceDate': 'ds', 'TotalSales': 'y'}, inplace=True)
    
    print("✅ Data prepared for Prophet successfully!")
    print("Daily sales data head:")
    print(daily_sales.head())

except FileNotFoundError:
    print(f"❌ Error: The file '{file_name}' was not found.")
    exit()

# --- Step 2: Train the Forecasting Model ---
# Initialize the Prophet model
model = Prophet()

# Fit the model with our daily sales data
model.fit(daily_sales)
print("\n✅ Prophet model trained successfully!")

# --- Step 3: Make Future Predictions ---
# Create a dataframe for future dates (we'll predict the next 90 days)
future = model.make_future_dataframe(periods=90)

# Make the forecast
forecast = model.predict(future)
print("\n✅ Forecast generated successfully!")

# --- Step 4: Display the Forecast ---
print("\n----------- Sales Forecast for the Next 90 Days -----------")
# Show the last 10 rows of the forecast, which will include our predictions
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
# --- Step 5: Visualize the Forecast ---
from prophet.plot import plot
import matplotlib.pyplot as plt

print("\n----------- Generating Forecast Plot -----------")
# Use Prophet's built-in plotting function
fig = model.plot(forecast)
plt.title('Sales Forecast for the Next 90 Days')
plt.xlabel('Date')
plt.ylabel('Sales')

# Save the plot to a file
plot_filename = 'sales_forecast.png'
fig.savefig(plot_filename)

print(f"\n✅ Forecast plot has been saved as '{plot_filename}'")
print("Check your project folder to see the image!")