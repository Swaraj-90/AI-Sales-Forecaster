import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import plotly.express as px
import datetime as dt
from pytrends.request import TrendReq

# --- Page Setup ---
st.set_page_config(page_title="AI Business Simulator", layout="wide")
st.title("🚀 Next-Level AI Business Simulator")

# --- Data Loading & Caching ---
@st.cache_data
def load_and_clean_data(file_name='online_retail.csv'):
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

df = load_and_clean_data()

# --- Sidebar ---
st.sidebar.header("⚙️ Controls")
st.sidebar.markdown("---")
st.sidebar.markdown("""
## 📖 About
This dashboard is an AI-powered business simulator that forecasts sales, analyzes customer behavior, integrates real-world Google Trends data, and helps optimize pricing strategies.
""")
top_products = df['Description'].value_counts().nlargest(20).index.tolist()
selected_product = st.sidebar.selectbox("Select a Product", options=top_products)
product_df = df[df['Description'] == selected_product]

# --- Main Page ---
st.header(f"Analysis for: {selected_product}")

# --- Key Metrics ---
total_sales = (product_df['Quantity'] * product_df['UnitPrice']).sum()
avg_price = product_df['UnitPrice'].mean()
total_transactions = product_df['InvoiceNo'].nunique()
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"£{total_sales:,.2f}")
col2.metric("Average Price", f"£{avg_price:,.2f}")
col3.metric("Total Transactions", f"{total_transactions}")

# --- Caching All Model Functions ---
@st.cache_data
def fetch_google_trends(keyword):
    pytrends = TrendReq(hl='en-US', tz=360)
    search_term = " ".join(keyword.split()[:2])
    try:
        pytrends.build_payload([search_term], cat=0, timeframe='all', geo='', gprop='')
        trends_df = pytrends.interest_over_time()
        if not trends_df.empty:
            trends_df = trends_df.reset_index().rename(columns={'date': 'ds', search_term: 'trend_strength'})
            return trends_df[['ds', 'trend_strength']]
    except Exception as e:
        return None
    return None

@st.cache_data
def create_forecast(_product_df, _trends_data):
    _product_df['TotalSales'] = _product_df['Quantity'] * _product_df['UnitPrice']
    daily_sales = _product_df.set_index('InvoiceDate').resample('D')['TotalSales'].sum().reset_index()
    daily_sales.rename(columns={'InvoiceDate': 'ds', 'TotalSales': 'y'}, inplace=True)
    model = Prophet()
    if _trends_data is not None:
        daily_sales = pd.merge(daily_sales, _trends_data, on='ds', how='left').fillna(0)
        model.add_regressor('trend_strength')
    model.fit(daily_sales)
    future = model.make_future_dataframe(periods=90)
    if _trends_data is not None:
        future = pd.merge(future, _trends_data, on='ds', how='left').fillna(0)
    forecast = model.predict(future)
    return model, forecast

@st.cache_data
def perform_segmentation(_df):
    snapshot_date = _df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = _df.groupby('CustomerID').agg({'InvoiceDate': lambda date: (snapshot_date - date.max()).days, 'InvoiceNo': 'nunique', 'UnitPrice': 'sum'})
    rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'UnitPrice': 'MonetaryValue'}, inplace=True)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10).fit(rfm_scaled)
    rfm['Cluster'] = kmeans.labels_
    return rfm

@st.cache_data
def train_price_model(_product_df):
    price_df = _product_df[['UnitPrice', 'Quantity']].copy()
    price_df['log_price'] = np.log1p(price_df['UnitPrice'])
    price_df['log_quantity'] = np.log1p(price_df['Quantity'])
    X = price_df[['log_price']]
    y = price_df['log_quantity']
    model = LinearRegression()
    model.fit(X, y)
    return model

# --- Run Models ---
trends_data = fetch_google_trends(selected_product)
model_prophet, forecast = create_forecast(product_df, trends_data)
customer_segments = perform_segmentation(df)
model_price = train_price_model(product_df)

# --- Tabs for Organization ---
tab_list = ["⚖️ Pricing Simulator", "📊 Sales Forecast", "🤖 AI Explanations", "👥 Customer Segments", "🌍 Google Trends", "📋 Raw Data"]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)

with tab1:
    st.header("Dynamic Pricing Simulator")
    st.write("See how changing the price might affect sales and revenue.")
    current_price = product_df['UnitPrice'].median()
    new_price = st.slider("Select a new price:", min_value=float(current_price * 0.5), max_value=float(current_price * 2), value=float(current_price), step=0.01, format="£%.2f")
    log_new_price = np.log1p(new_price)
    predicted_log_quantity = model_price.predict([[log_new_price]])
    predicted_quantity = np.expm1(predicted_log_quantity)[0]
    predicted_revenue = new_price * predicted_quantity
    st.subheader("Predicted Outcome")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Sales (Units)", f"{predicted_quantity:,.0f} units")
    col2.metric("Predicted Revenue", f"£{predicted_revenue:,.2f}")

with tab2:
    st.subheader("Sales Forecast Visualization")
    fig1 = model_prophet.plot(forecast)
    st.pyplot(fig1)

with tab3:
    st.subheader("Explainable AI: Forecast Components")
    fig2 = model_prophet.plot_components(forecast)
    st.pyplot(fig2)

with tab4:
    st.subheader("Customer Segments")
    fig3 = px.scatter_3d(customer_segments, x='Recency', y='Frequency', z='MonetaryValue', color='Cluster', hover_data=['Cluster'])
    st.plotly_chart(fig3, use_container_width=True)

with tab5:
    st.subheader("Google Trends Analysis")
    if trends_data is not None:
        search_term = " ".join(selected_product.split()[:2])
        st.write(f"Showing Google Trends data for the search term: '{search_term}'")
        st.line_chart(trends_data.set_index('ds'))
    else:
        st.warning("Could not retrieve Google Trends data for this product.")

with tab6:
    st.subheader("Raw Forecast Data")
    with st.expander("Click to view"):
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(90))