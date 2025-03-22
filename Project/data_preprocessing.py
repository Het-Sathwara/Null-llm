import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import os

# Create necessary directories matching the training script path
os.makedirs("Project/data/raw", exist_ok=True)
os.makedirs("Project/data/processed", exist_ok=True)
os.makedirs("Project/models", exist_ok=True)

# Fetch AAPL data
ticker = "AAPL"
start_date = "2012-01-01"
end_date = "2019-12-31"

print(f"\nDownloading {ticker} data from {start_date} to {end_date}...")
stock = yf.Ticker(ticker)
data = stock.history(start=start_date, end=end_date, interval="1d")

print("\nAdding technical indicators...")
# Keep only OHLCV columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Add 20-day SMA
data['SMA_20'] = ta.sma(data['Close'], length=20)

# Add 14-day RSI
data['RSI_14'] = ta.rsi(data['Close'], length=14)

# Add MACD
macd = ta.macd(data['Close'])
data = pd.concat([data, macd], axis=1)

# Add Bollinger Bands
bbands = ta.bbands(data['Close'], length=20)
data = pd.concat([data, bbands], axis=1)

# Drop NaN rows (indicators generate NaNs initially)
data.dropna(inplace=True)

# Convert timezone-aware dates to timezone-naive dates
data.index = data.index.tz_localize(None)

print("\nData summary:")
print(data.info())
print("\nFirst few rows:")
print(data.head())
print("\nDate range:")
print(f"Start date: {data.index.min()}")
print(f"End date: {data.index.max()}")

print("\nSaving raw data...")
# Save raw data with indicators (preserve index)
data.to_csv("Project/data/raw/aapl_with_indicators.csv", date_format='%Y-%m-%d')

print("\nProcessing data...")
# All columns except date index should be normalized
cols_to_normalize = data.columns.tolist()
data_to_scale = data[cols_to_normalize]
scaler = MinMaxScaler()

# Split into training (2012–2017) and testing (2018–2019)
split_date = pd.Timestamp('2017-12-31')
train_data = data_to_scale[data_to_scale.index <= split_date]
test_data = data_to_scale[data_to_scale.index > split_date]

print("\nSplit information:")
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Fit scaler on training data only
scaler.fit(train_data)

# Scale training data
train_scaled = scaler.transform(train_data)
train_scaled_df = pd.DataFrame(train_scaled, columns=cols_to_normalize, index=train_data.index)

# Scale test data using the SAME scaler
test_scaled = scaler.transform(test_data)
test_scaled_df = pd.DataFrame(test_scaled, columns=cols_to_normalize, index=test_data.index)

print("\nSaving processed data...")
# Save processed data matching the paths in training script
train_scaled_df.to_csv("Project/data/processed/train_scaled.csv", date_format='%Y-%m-%d')
test_scaled_df.to_csv("Project/data/processed/test_scaled.csv", date_format='%Y-%m-%d')

print("\nFirst few rows of scaled training data:")
print(train_scaled_df.head())

print("\nData preparation completed successfully!")