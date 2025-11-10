import os
import io
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# =======================
# 1️⃣ LOAD KONFIGURASI
# =======================
load_dotenv()
API_KEYS = os.getenv("ALPHA_VANTAGE_KEY", "").split(",")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not API_KEYS or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("❌ Pastikan .env memiliki ALPHA_VANTAGE_KEY, TELEGRAM_TOKEN, dan TELEGRAM_CHAT_ID")

# =======================
# 2️⃣ PENGATURAN DASAR
# =======================
PAIR_LIST = ["EURUSD", "GBPUSD", "USDJPY"]
API_URL = "https://www.alphavantage.co/query"
INTERVAL = "5min"
PREDICT_INTERVAL = 30  # MODE AGRESIF: prediksi setiap 30 detik
API_INDEX = 0
TRAIN_REFRESH = 6 * 60 * 60  # retrain setiap 6 jam

# =======================
# 3️⃣ UTILITAS TELEGRAM
# =======================
def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"⚠️ Gagal kirim pesan ke Telegram: {e}")

def send_telegram_photo(image_bytes, caption: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        files = {"photo": image_bytes}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"}
        requests.post(url, data=data, files=files)
    except Exception as e:
        print(f"⚠️ Gagal kirim foto ke Telegram: {e}")

# =======================
# 4️⃣ FUNGSI DATA
# =======================
def get_forex_data(pair: str):
    """Ambil data dari AlphaVantage (rotasi API + retry)"""
    global API_INDEX
    for _ in range(len(API_KEYS)):
        api_key = API_KEYS[API_INDEX % len(API_KEYS)]
        API_INDEX += 1
        print(f"📊 Mengambil data {pair} dengan API {api_key} ...")

        params = {
            "function": "FX_INTRADAY",
            "from_symbol": pair[:3],
            "to_symbol": pair[3:],
            "interval": INTERVAL,
            "apikey": api_key,
            "datatype": "csv"
        }

        try:
            response = requests.get(API_URL, params=params)
            if "Thank you" in response.text:
                print(f"⚠️ API {api_key} limit. Ganti API...")
                continue

            df = pd.read_csv(StringIO(response.text))
            if "close" not in df.columns:
                print(f"⚠️ Data API {api_key} tidak memiliki kolom 'close'.")
                continue

            df = df.sort_values("timestamp")
            return df[["timestamp", "close"]]
        except Exception as e:
            print(f"⚠️ Error API {api_key}: {e}")
            continue

    print("😴 Semua API limit. Tunggu 2 menit lalu coba lagi...")
    time.sleep(120)
    return get_forex_data(pair)

# =======================
# 5️⃣ MODEL LSTM
# =======================
def train_lstm_model(df):
    data = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model, scaler

def predict_trend(model, scaler, df):
    data = df["close"].values.reshape(-1, 1)
    scaled = scaler.transform(data)
    X_test = np.array([scaled[-60:]])
    pred = model.predict(X_test)
    return scaler.inverse_transform(pred)[0][0], df["close"].iloc[-1]

def generate_chart(df, pair, pred_price, last_price, action):
    plt.figure(figsize=(8, 4))
    plt.plot(df["timestamp"].tail(50), df["close"].tail(50), label="Harga Aktual", linewidth=2)
    plt.axhline(y=pred_price, color="green" if action == "BUY" else "red", linestyle="--", label="Prediksi AI")
    plt.xticks(rotation=45)
    plt.title(f"{pair} - Tren AI ({action})")
    plt.xlabel("Waktu")
    plt.ylabel("Harga")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# =======================
# 6️⃣ LOOP UTAMA
# =======================
print("🚀 Menjalankan AI Forex Bot - MODE AGRESIF (Prediksi setiap 30 detik)...")

last_train_time = time.time()
models = {}

while True:
    # retrain model tiap 6 jam
    if time.time() - last_train_time > TRAIN_REFRESH or not models:
        print("🧠 Melatih ulang model AI...")
        for pair in PAIR_LIST:
            df = get_forex_data(pair)
            if df is None or df.empty:
                continue
            model, scaler = train_lstm_model(df)
            models[pair] = (model, scaler)
        last_train_time = time.time()
        print("✅ Model AI selesai dilatih.")

    # prediksi tiap 30 detik
    for pair in PAIR_LIST:
        df = get_forex_data(pair)
        if df is None or df.empty:
            continue

        model, scaler = models.get(pair, (None, None))
        if model is None:
            continue

        pred_price, last_price = predict_trend(model, scaler, df)
        diff = ((pred_price - last_price) / last_price) * 100
        action = "BUY" if diff > 0 else "SELL"
        emoji = "🔵" if action == "BUY" else "🔴"
        waktu = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        caption = (
            f"{emoji} *Sinyal AI Forex ({pair})*\n"
            f"📅 {waktu}\n"
            f"💹 Aksi: *{action}*\n"
            f"💰 Harga Sekarang: {last_price:.5f}\n"
            f"🎯 Prediksi AI: {pred_price:.5f}\n"
            f"📈 Perubahan: {diff:.3f}%"
        )

        chart = generate_chart(df, pair, pred_price, last_price, action)
        send_telegram_photo(chart, caption)

        print(f"✅ {pair}: {action} | Δ {diff:.3f}% | {waktu}")
        time.sleep(PREDICT_INTERVAL)