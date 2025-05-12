# app.py
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, Response
from datetime import datetime, timedelta
import os
import logging
import matplotlib.pyplot as plt
import io
import time  # Add this import
import sqlite3  # Add this import

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for stock data to reduce API calls
stock_cache = {}
last_fetch_time = {}

# List of top 10 US companies by market cap (as of 2025)
COMPANIES = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'NVDA': 'Nvidia',
    'GOOGL': 'Alphabet',
    'AMZN': 'Amazon',
    'META': 'Meta',
    'TSLA': 'Tesla',
    'BRK-B': 'Berkshire Hathaway',
    'LLY': 'Eli Lilly',
    'AVGO': 'Broadcom'
}

# Initialize SQLite database for persistent caching
def init_db():
    conn = sqlite3.connect('stock_cache.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS stock_data (
                        symbol TEXT,
                        start_date TEXT,
                        end_date TEXT,
                        data TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, start_date, end_date))''')
    conn.commit()
    conn.close()

# Save data to SQLite cache
def save_to_db(symbol, start_date, end_date, data):
    conn = sqlite3.connect('stock_cache.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT OR REPLACE INTO stock_data (symbol, start_date, end_date, data) 
                      VALUES (?, ?, ?, ?)''', (symbol, start_date, end_date, data))
    conn.commit()
    conn.close()

# Load data from SQLite cache
def load_from_db(symbol, start_date, end_date):
    conn = sqlite3.connect('stock_cache.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT data FROM stock_data WHERE symbol = ? AND start_date = ? AND end_date = ?''',
                   (symbol, start_date, end_date))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

# Function to fetch stock data from Yahoo Finance with enhanced caching
# and reduced API calls to avoid hitting the yfinance limit
def fetch_stock_data(symbol, start_date, end_date, retries=3):
    current_time = datetime.now()

    # Check SQLite cache
    cached_data = load_from_db(symbol, start_date, end_date)
    if cached_data:
        logger.info(f"Returning cached data from database for {symbol}")
        return pd.read_json(cached_data)

    attempt = 0
    backoff_time = 5  # Initial backoff time in seconds
    while attempt < retries:
        try:
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data found for {symbol}")

            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            if not validate_data(df, symbol):
                raise ValueError(f"Invalid data for {symbol}: contains zero or negative values")

            # Save to SQLite cache
            save_to_db(symbol, start_date, end_date, df.to_json())

            logger.info(f"Successfully fetched data for {symbol}")
            return df
        except Exception as e:
            attempt += 1
            logger.error(f"Attempt {attempt} failed for {symbol}: {e}")
            if attempt < retries:
                logger.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff

    logger.error(f"Failed to fetch data for {symbol} after {retries} attempts")

    # Return placeholder data if no cache is available
    logger.warning(f"Returning placeholder data for {symbol}")
    placeholder_data = pd.DataFrame({
        'Date': pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d %H:%M:%S'),
        'Open': [0] * len(pd.date_range(start=start_date, end=end_date, freq='D')),
        'High': [0] * len(pd.date_range(start=start_date, end=end_date, freq='D')),
        'Low': [0] * len(pd.date_range(start=start_date, end=end_date, freq='D')),
        'Close': [0] * len(pd.date_range(start=start_date, end=end_date, freq='D')),
        'Volume': [0] * len(pd.date_range(start=start_date, end=end_date, freq='D'))
    })
    return placeholder_data

# Function to validate fetched data
def validate_data(df, symbol):
    if df.empty:
        return False
    if (df['Open'] <= 0).any() or (df['Close'] <= 0).any() or (df['Volume'] <= 0).any():
        logger.error(f"Validation failed for {symbol}: Found zero or negative values in Open, Close, or Volume")
        return False
    return True

# Function to check for significant price changes
def check_price_change(df, symbol, threshold=5):
    if len(df) < 2:
        return None
    latest_price = df['Close'].iloc[-1]
    previous_price = df['Close'].iloc[-2]
    percentage_change = ((latest_price - previous_price) / previous_price) * 100
    if abs(percentage_change) >= threshold:
        return f"Alert: {symbol} price changed by {percentage_change:.2f}% (Latest: ${latest_price:.2f})"
    return None

# Route for the homepage
@app.route('/')
def index():
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    df = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if df.empty:
        return render_template('index.html', error=f"Failed to load data for {symbol}. Please try again later.", companies=COMPANIES)

    notification = check_price_change(df, symbol)
    
    # Convert DataFrame to HTML for display
    hist_data_html = df.to_html(classes='table table-striped', index=False)
    
    return render_template('index.html',
                         hist_data=hist_data_html,
                         notification=notification,
                         default_symbol=symbol,
                         default_start=start_date.strftime('%Y-%m-%d'),
                         default_end=end_date.strftime('%Y-%m-%d'),
                         companies=COMPANIES)

# Route for updating data
@app.route('/update', methods=['POST'])
def update():
    symbol = request.form.get('company', 'AAPL')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    if not symbol or not start_date or not end_date:
        logger.error("Missing parameters in the update request.")
        return jsonify({'error': "Company, start date, and end date are required."}), 400

    df = fetch_stock_data(symbol, start_date, end_date)
    if df.empty:
        logger.error(f"Failed to load data for {symbol}.")
        return jsonify({'error': f"Failed to load data for {symbol}. Please try again later."}), 500

    notification = check_price_change(df, symbol)

    hist_data_html = df.to_html(classes='table table-striped', index=False)

    return jsonify({
        'hist_data': hist_data_html,
        'notification': notification
    })

# Route for exporting data as CSV
@app.route('/export_csv', methods=['POST'])
def export_csv():
    symbol = request.form.get('company', 'AAPL')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    df = fetch_stock_data(symbol, start_date, end_date)
    if df.empty:
        return "Failed to export data", 500
    
    csv_path = f"{symbol}_data.csv"
    df.to_csv(csv_path, index=False)
    
    return send_file(csv_path, as_attachment=True, download_name=f"{symbol}_stock_data.csv")

# Route for generating and serving a stock price graph
@app.route('/plot/<symbol>')
def plot(symbol):
    start_date = request.args.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))

    if not symbol:
        logger.error("Symbol is missing in the request.")
        return "Symbol is required to generate a plot", 400

    df = fetch_stock_data(symbol, start_date, end_date)
    if df.empty:
        logger.error(f"No data available for {symbol} to plot.")
        return "No data available to plot", 404

    # Create the plot
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA270'] = df['Close'].rolling(window=270).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['Date'], df['MA30'], label='30-Day MA', color='orange', linestyle='--')
    plt.plot(df['Date'], df['MA270'], label='270-Day MA', color='green', linestyle='--')
    plt.title(f"{COMPANIES.get(symbol, symbol)} Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return Response(img, mimetype='image/png')

# Initialize the database
init_db()

if __name__ == '__main__':
    app.run(debug=True)
