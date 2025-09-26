# predictor.py
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime,timedelta
from joblib import load
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from SAmodel import utils

PRICE_MODEL_PATH = "SAmodel/price_model/xgb_price_model.joblib"
SENT_MODEL_PATH = "SAmodel/sentiment_model"
CSV_PATH = "alert_log.csv"

NEWS_API_KEY = "0a22dd9afa05422289935d324470a150"

TICKERS = [
    "AAPL", "MSFT", "TSLA", "GOOG", "AMZN",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "WIPRO.NS", "BAJAJ-AUTO.NS", "ITC.NS", "ADANIPORTS.NS"
]

def fetch_price_latest(ticker):
    df = yf.download(ticker, period="5d", progress=False)
    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low",
        "Close":"close","Adj Close":"adj_close","Volume":"volume"
    })
    return df





def fetch_latest_news(ticker, limit=1):
    base_url = "https://newsapi.org/v2/everything"
    query = ticker.replace(".NS", "")
    
    params = {
        "q": query,
        "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "pageSize": limit,
        "language": "en",
        "apiKey": NEWS_API_KEY
    }
    try:
        resp = requests.get(base_url, params=params)
        data = resp.json()
        if "articles" in data and len(data["articles"]) > 0:
            return data["articles"][0]["title"]
    
        params["from"] = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        resp = requests.get(base_url, params=params)
        data = resp.json()
        if "articles" in data and len(data["articles"]) > 0:
            return data["articles"][0]["title"]
        
        return "No news found"
    except Exception as e:
        return f"News fetch error: {e}"

def prepare_features(df, sentiment_score):
    df = df.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_2"] = df["close"].pct_change(2)
    df["ma_3"] = df["close"].rolling(3).mean()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["volatility_5"] = df["ret_1"].rolling(5).std()
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(5).mean()+1e-9)
    df = df.fillna(0)

    last = df.iloc[-1]
    features = {
        "ret_1": float(last["ret_1"]),
        "ret_2": float(last["ret_2"]),
        "ma_3": float(last["ma_3"]),
        "ma_5": float(last["ma_5"]),
        "volatility_5": float(last["volatility_5"]),
        "vol_ratio": float(last["vol_ratio"]),
        "sentiment": float(sentiment_score),
        "sent_sentma_3": float(sentiment_score)
    }
    return pd.DataFrame([features]), float(last["close"])

def predict_ticker(ticker, price_model, sent_pipe):
    df = fetch_price_latest(ticker)
    news = fetch_latest_news(ticker)

    preds = sent_pipe([news], truncation=True)[0]
    sentiment_score = utils.sentiment_to_score(preds)

    X, current_price = prepare_features(df, sentiment_score)

    proba = price_model.predict_proba(X)[0]
    pred = int(price_model.predict(X)[0])
    predicted_price = current_price * (1 + (proba[1] - proba[0]) * 0.02)

    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "prediction": "UP" if pred == 1 else "DOWN",
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "change_abs": abs(predicted_price - current_price),
        "news": news
    }

if __name__ == "__main__":
    price_model = load(PRICE_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_PATH)
    sent_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    results = []
    for t in TICKERS:
        try:
            res = predict_ticker(t, price_model, sent_pipe)
            results.append(res)
        except Exception as e:
            print(f"Error for {t}: {e}")

    results = sorted(results, key=lambda x: x["change_abs"], reverse=True)[:7]
    df_out = pd.DataFrame(results).drop(columns=["change_abs"])

    df_out.to_csv(CSV_PATH, mode="a", header=not pd.io.common.file_exists(CSV_PATH), index=False)

    print("✅ Predictions saved to CSV")
