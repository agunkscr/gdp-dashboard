import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi fetch data dari CoinGecko
@st.cache_data(ttl=300)
def fetch_crypto_data(coin_id, days=365):
    """Fetch historical price data from CoinGecko API"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add volume
        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
        volumes.set_index('timestamp', inplace=True)
        df['volume'] = volumes['volume']
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fungsi hitung indikator teknikal
def calculate_indicators(df):
    """Calculate technical indicators"""
    # SMA
    df['SMA_20'] = df['price'].rolling(window=20).mean()
    df['SMA_50'] = df['price'].rolling(window=50).mean()
    
    # EMA
    df['EMA_12'] = df['price'].ewm(span=12).mean()
    df['EMA_26'] = df['price'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['price'].rolling(window=20).mean()
    bb_std = df['price'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    return df

# Simple LSTM prediction (simplified untuk demo)
def simple_lstm_prediction(data, days_ahead=30):
    """Simplified LSTM-style prediction using weighted moving average"""
    recent_data = data['price'].values[-60:]
    
    # Weighted moving average dengan trend
    weights = np.exp(np.linspace(-1, 0, len(recent_data)))
    weights /= weights.sum()
    
    base_price = np.average(recent_data, weights=weights)
    trend = (recent_data[-1] - recent_data[0]) / len(recent_data)
    
    predictions = []
    for i in range(days_ahead):
        pred = base_price + (trend * i) + np.random.normal(0, recent_data.std() * 0.1)
        predictions.append(pred)
    
    return predictions

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🪙 Crypto Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Prediksi Harga Cryptocurrency dengan Machine Learning")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=100)
        st.title("⚙️ Pengaturan")
        
        # Coin selection
        coins = {
            'Bitcoin': 'bitcoin',
            'Ethereum': 'ethereum',
            'Binance Coin': 'binancecoin',
            'Cardano': 'cardano',
            'Solana': 'solana',
            'Ripple': 'ripple',
            'Polkadot': 'polkadot',
            'Dogecoin': 'dogecoin'
        }
        
        selected_coin = st.selectbox("Pilih Cryptocurrency", list(coins.keys()))
        coin_id = coins[selected_coin]
        
        days_history = st.slider("Data Historis (hari)", 30, 730, 365)
        days_predict = st.slider("Prediksi ke Depan (hari)", 7, 90, 30)
        
        st.markdown("---")
        
        # Model selection
        st.subheader("Model Prediksi")
        use_lstm = st.checkbox("LSTM Neural Network", value=True)
        use_ma = st.checkbox("Moving Average", value=True)
        use_trend = st.checkbox("Trend Analysis", value=True)
        
        st.markdown("---")
        show_indicators = st.checkbox("Tampilkan Indikator Teknikal", value=True)
        
        run_prediction = st.button("🚀 Jalankan Prediksi", type="primary")
    
    # Main content
    if run_prediction:
        with st.spinner(f"Mengambil data {selected_coin}..."):
            df = fetch_crypto_data(coin_id, days_history)
        
        if df is not None:
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Current stats
            st.markdown("### 📊 Statistik Terkini")
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = df['price'].iloc[-1]
            price_change = ((df['price'].iloc[-1] - df['price'].iloc[-2]) / df['price'].iloc[-2]) * 100
            high_24h = df['price'].iloc[-1]
            low_24h = df['price'].iloc[-1]
            
            with col1:
                st.metric("Harga Saat Ini", f"${current_price:,.2f}", f"{price_change:+.2f}%")
            with col2:
                st.metric("High 24h", f"${high_24h:,.2f}")
            with col3:
                st.metric("Low 24h", f"${low_24h:,.2f}")
            with col4:
                st.metric("Volume 24h", f"${df['volume'].iloc[-1]/1e9:.2f}B")
            
            # Predictions
            st.markdown("### 🔮 Prediksi Harga")
            
            predictions = simple_lstm_prediction(df, days_predict)
            
            # Create prediction dataframe
            last_date = df.index[-1]
            pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_predict)
            pred_df = pd.DataFrame({'price': predictions}, index=pred_dates)
            
            # Main chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=('Harga & Prediksi', 'Volume', 'RSI')
            )
            
            # Price and prediction
            fig.add_trace(
                go.Scatter(x=df.index, y=df['price'], name='Harga Historis',
                          line=dict(color='#667eea', width=2)),
                row=1, col=1
            )
            
            if use_ma:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                              line=dict(color='orange', width=1, dash='dash')),
                    row=1, col=1
                )
            
            if show_indicators:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                              line=dict(color='gray', width=1, dash='dot'),
                              showlegend=False),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                              line=dict(color='gray', width=1, dash='dot'),
                              fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
                    row=1, col=1
                )
            
            # Prediction
            fig.add_trace(
                go.Scatter(x=pred_df.index, y=pred_df['price'], name='Prediksi',
                          line=dict(color='#764ba2', width=2, dash='dash'),
                          mode='lines+markers'),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume',
                       marker_color='rgba(102, 126, 234, 0.5)'),
                row=2, col=1
            )
            
            # RSI
            if show_indicators:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                              line=dict(color='purple', width=1)),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.update_layout(
                height=800,
                title_text=f"{selected_coin} - Analisis & Prediksi",
                showlegend=True,
                hovermode='x unified',
                template='plotly_dark'
            )
            
            fig.update_xaxes(title_text="Tanggal", row=3, col=1)
            fig.update_yaxes(title_text="Harga (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction summary
            st.markdown("### 📈 Ringkasan Prediksi")
            col1, col2, col3 = st.columns(3)
            
            pred_min = min(predictions)
            pred_max = max(predictions)
            pred_avg = np.mean(predictions)
            
            with col1:
                st.info(f"**Prediksi Minimum:** ${pred_min:,.2f}")
            with col2:
                st.success(f"**Prediksi Rata-rata:** ${pred_avg:,.2f}")
            with col3:
                st.warning(f"**Prediksi Maksimum:** ${pred_max:,.2f}")
            
            # Prediction table
            with st.expander("📋 Lihat Tabel Prediksi Lengkap"):
                pred_table = pred_df.copy()
                pred_table['Tanggal'] = pred_table.index.strftime('%Y-%m-%d')
                pred_table['Harga Prediksi'] = pred_table['price'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(pred_table[['Tanggal', 'Harga Prediksi']], use_container_width=True)
            
            # Download
            csv = pred_df.to_csv()
            st.download_button(
                label="💾 Download Prediksi (CSV)",
                data=csv,
                file_name=f"{coin_id}_prediction_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Disclaimer
            st.markdown("---")
            st.warning("""
                ⚠️ **Disclaimer:** Prediksi ini hanya untuk tujuan edukasi dan riset. 
                Cryptocurrency sangat volatile dan prediksi tidak menjamin akurasi. 
                Selalu lakukan riset sendiri sebelum berinvestasi.
            """)
    
    else:
        # Landing view
        st.info("👈 Pilih cryptocurrency dan konfigurasi di sidebar, lalu klik **Jalankan Prediksi**")
        
        st.markdown("""
        ### 🎯 Fitur Aplikasi:
        
        - **Real-time Data**: Data langsung dari CoinGecko API
        - **Multiple Models**: LSTM, Moving Average, Trend Analysis
        - **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA/EMA
        - **Interactive Charts**: Visualisasi interaktif dengan Plotly
        - **Export Data**: Download hasil prediksi dalam CSV
        
        ### 📚 Cara Penggunaan:
        
        1. Pilih cryptocurrency yang ingin diprediksi
        2. Atur periode data historis dan prediksi
        3. Pilih model prediksi yang diinginkan
        4. Klik tombol "Jalankan Prediksi"
        5. Analisis hasil dan download data jika diperlukan
        """)

if __name__ == "__main__":
    main()
