# 🪙 Crypto Price Predictor

Aplikasi prediksi harga cryptocurrency profesional menggunakan Machine Learning dengan antarmuka Streamlit yang intuitif.

## ✨ Fitur Utama

- **Multi-Model Prediction**: LSTM, Prophet, ARIMA, dan XGBoost
- **Real-time Data**: Integrasi dengan API CoinGecko
- **Analisis Teknikal**: RSI, MACD, Bollinger Bands, Moving Averages
- **Visualisasi Interaktif**: Grafik candlestick, volume, dan prediksi
- **Backtesting**: Evaluasi performa model dengan data historis
- **Export Data**: Download hasil prediksi dalam format CSV

## 🚀 Quick Start

### Instalasi

```bash
# Clone repository
git clone https://github.com/username/crypto-price-predictor.git
cd crypto-price-predictor

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur Direktori

```
crypto-price-predictor/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # Documentation
│
├── models/               # ML models
│   ├── lstm_model.py
│   ├── prophet_model.py
│   ├── arima_model.py
│   └── xgboost_model.py
│
├── utils/                # Utility functions
│   ├── data_fetcher.py   # API integration
│   ├── indicators.py     # Technical indicators
│   ├── preprocessor.py   # Data preprocessing
│   └── visualizer.py     # Plotting functions
│
├── config/               # Configuration
│   └── settings.py       # App settings
│
└── data/                 # Data storage (generated)
    ├── cache/
    └── exports/
```

## 🛠️ Teknologi

- **Python 3.8+**
- **Streamlit**: Web framework
- **TensorFlow/Keras**: Deep learning (LSTM)
- **Prophet**: Time series forecasting
- **statsmodels**: ARIMA modeling
- **XGBoost**: Gradient boosting
- **pandas & numpy**: Data manipulation
- **plotly**: Interactive visualization
- **requests**: API calls

## 📊 Model yang Tersedia

### 1. LSTM (Long Short-Term Memory)
Deep learning model untuk menangkap pola temporal kompleks dalam data time series.

### 2. Prophet
Model forecasting dari Facebook yang robust terhadap missing data dan seasonal patterns.

### 3. ARIMA
Model statistik klasik untuk time series stasioner.

### 4. XGBoost
Ensemble learning dengan gradient boosting untuk prediksi berdasarkan fitur teknikal.

## 🎯 Cara Penggunaan

1. **Pilih Cryptocurrency**: Pilih dari daftar coin populer (BTC, ETH, dll)
2. **Set Time Range**: Tentukan periode data historis
3. **Pilih Model**: Pilih satu atau beberapa model prediksi
4. **Configure Parameters**: Sesuaikan hyperparameter model
5. **Run Prediction**: Klik tombol untuk mulai prediksi
6. **Analyze Results**: Lihat visualisasi dan metrik evaluasi

## 📈 Indikator Teknikal

- **SMA/EMA**: Simple & Exponential Moving Average
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands
- **Volume Analysis**: Trading volume patterns

## ⚙️ Konfigurasi

Edit `config/settings.py` untuk menyesuaikan:

```python
# API Configuration
COINGECKO_API = "https://api.coingecko.com/api/v3"
CACHE_DURATION = 300  # seconds

# Model Parameters
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
PROPHET_CHANGEPOINT_PRIOR = 0.05

# UI Settings
DEFAULT_COIN = "bitcoin"
DEFAULT_DAYS = 365
```

## 📝 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
tensorflow>=2.13.0
prophet>=1.1.0
statsmodels>=0.14.0
xgboost>=2.0.0
plotly>=5.17.0
requests>=2.31.0
scikit-learn>=1.3.0
ta>=0.11.0
```

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. Fork repository
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## ⚠️ Disclaimer

**Aplikasi ini hanya untuk tujuan edukasi dan riset.** Prediksi cryptocurrency sangat volatile dan tidak dijamin akurat. Jangan gunakan aplikasi ini sebagai satu-satunya dasar untuk keputusan investasi. Selalu lakukan riset sendiri (DYOR) dan konsultasi dengan ahli finansial.

## 📄 Lisensi

MIT License - lihat file `LICENSE` untuk detail.

## 👤 Author

Nama Kamu - [@username](https://github.com/username)

## 🙏 Acknowledgments

- CoinGecko untuk API gratis
- Streamlit community
- Open source ML libraries

---

**⭐ Jangan lupa beri star jika project ini bermanfaat!**
