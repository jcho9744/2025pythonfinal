# app.py
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, Response
from datetime import datetime, timedelta
import os
import logging
import matplotlib.pyplot as plt
import io

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

# Function to fetch stock data from Yahoo Finance with retry logic
def fetch_stock_data(symbol, start_date, end_date, retries=3):
    current_time = datetime.now()
    cache_key = f"{symbol}_{start_date}_{end_date}"
    
    if cache_key in stock_cache and (current_time - last_fetch_time.get(cache_key, current_time)).total_seconds() < 300:
        logger.info(f"Returning cached data for {symbol}")
        return stock_cache[cache_key]

    attempt = 0
    while attempt < retries:
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data found for {symbol}")
            
            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            if not validate_data(df, symbol):
                raise ValueError(f"Invalid data for {symbol}: contains zero or negative values")
            
            stock_cache[cache_key] = df
            last_fetch_time[cache_key] = current_time
            
            for key in list(stock_cache.keys()):
                if key.startswith(symbol) and key != cache_key:
                    del stock_cache[key]
                    del last_fetch_time[key]
            
            logger.info(f"Successfully fetched data for {symbol}")
            return df
        except Exception as e:
            attempt += 1
            logger.error(f"Attempt {attempt} failed for {symbol}: {e}")
            if attempt == retries:
                logger.error(f"Failed to fetch data for {symbol} after {retries} attempts")
                if cache_key in stock_cache:
                    del stock_cache[cache_key]
                return pd.DataFrame()

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
    
    df = fetch_stock_data(symbol, start_date, end_date)
    if df.empty:
        return jsonify({'error': f"Failed to load data for {symbol}. Please try again later."}), 500
    
    notification = check_price_change(df, symbol)
    
    # Convert DataFrame to HTML for AJAX update
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

    df = fetch_stock_data(symbol, start_date, end_date)
    if df.empty:
        return "No data available to plot", 404

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.title(f"{COMPANIES.get(symbol, symbol)} Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return Response(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
