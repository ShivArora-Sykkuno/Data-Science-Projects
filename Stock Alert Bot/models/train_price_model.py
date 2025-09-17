# train_price_model.py
"""
Train price model that uses historical OHLCV + sentiment features.
Saves model to SAmodel/price_model/xgb_price_model.joblib
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from SAmodel import utils
from SAmodel.price_model import save_price_model
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb

import joblib
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ------- CONFIG -------
TICKERS = ["AAPL", "MSFT", "TSLA"]   # Replace with tickers you want to train on
START = "2018-01-01"
END = datetime.today().strftime("%Y-%m-%d")
WINDOW = 5   # look-back window for features
SENT_NEWS_DAYS = 2  # aggregate news sentiment of last N days
MODEL_OUTPUT = "SAmodel/price_model/xgb_price_model.joblib"
SENT_MODEL_PATH = "SAmodel/sentiment_model"  # your fine-tuned sentiment model
# -----------------------

def fetch_price(ticker, start=START, end=END):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
    })
    df = df[["open","high","low","close","volume"]]
    return df

def make_technical_features(df):
    df = df.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_2"] = df["close"].pct_change(2)
    df["ma_3"] = df["close"].rolling(3).mean()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["volatility_5"] = df["ret_1"].rolling(5).std()
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(5).mean()+1e-9)
    df = df.fillna(0)
    return df

def attach_sentiment_feature(df, ticker, sentiment_pipeline, window_days=SENT_NEWS_DAYS):
    """
    Very simple: we will look in SAmodel/data/ for news files for the ticker (if you have them).
    Or if you don't, this function will compute sentiment=0 for all days.
    Idea: if you have a prebuilt CSV of news with columns [date, text] for each ticker, load and aggregate.
    For demo we assume there is file SAmodel/data/{ticker}_news.csv with 'date' and 'text' columns (date ISO).
    """
    df = df.copy()
    df["date"] = df.index.date
    sent_scores = pd.Series(0.0, index=df.index)

    news_path = f"SAmodel/data/{ticker}_news.csv"
    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path, parse_dates=["date"], encoding="latin-1", on_bad_lines="skip")
        news_df["date"] = news_df["date"].dt.date
        # for each day in df, collect news for last `window_days` and compute aggregated sentiment
        for d in df["date"].unique():
            startd = d - pd.Timedelta(days=window_days-1)
            mask = (news_df["date"] >= startd) & (news_df["date"] <= d)
            if mask.sum() == 0:
                sent_scores[df[df["date"]==d].index] = 0.0
            else:
                texts = news_df.loc[mask,"text"].astype(str).tolist()
                preds = sentiment_pipeline(texts, truncation=True)
                scores = [utils.sentiment_to_score(p) for p in preds]
                sent_scores[df[df["date"]==d].index] = np.mean(scores)
    else:
        # no local news -> zero sentiment
        sent_scores[:] = 0.0

    df["sentiment"] = sent_scores.values
    # add short rolling aggregation
    df["sent_sentma_3"] = df["sentiment"].rolling(3, min_periods=1).mean().fillna(0)
    return df

def construct_dataset(tickers):
    X_rows = []
    y_rows = []
    meta = []
    # load HF sentiment pipeline once
    tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_PATH)
    sent_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    for t in tickers:
        print("Fetching price:", t)
        df = fetch_price(t)
        if df.shape[0] < 20:
            print("Skipping - not enough data:", t)
            continue
        df = make_technical_features(df)
        df = attach_sentiment_feature(df, t, sent_pipe)
        # print("Raw columns:", df.columns.tolist())
        df = utils.compute_label_next_day(df, price_col="close")
        # drop last row because label is NaN for last day (next_close missing)
        df = df.dropna(subset=["label"])
        # build features - use recent WINDOW lags
        for idx in range(WINDOW, len(df)):
            window_df = df.iloc[idx-WINDOW:idx]
            # take last row features
            last = df.iloc[idx-1]
            features = {
                "ret_1": last["ret_1"],
                "ret_2": last["ret_2"],
                "ma_3": last["ma_3"],
                "ma_5": last["ma_5"],
                "volatility_5": last["volatility_5"],
                "vol_ratio": last["vol_ratio"],
                "sentiment": last["sentiment"],
                "sent_sentma_3": last["sent_sentma_3"]
            }
            X_rows.append(features)
            y_rows.append(int(df.iloc[idx]["label"]))
            meta.append({"ticker": t, "date": df.index[idx]})
    X = pd.DataFrame(X_rows)
    y = pd.Series(y_rows)
    meta_df = pd.DataFrame(meta)
    return X.fillna(0), y, meta_df

def main():
    X, y, meta = construct_dataset(TICKERS)
    print("X shape:", X.shape, "y dist:", y.value_counts().to_dict())

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
    print("Training XGBoost...")
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, use_label_encoder=False,
                          eval_metric="logloss", n_jobs=4)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    # save model
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    print("Saved model to", MODEL_OUTPUT)

if __name__ == "__main__":
    main()
