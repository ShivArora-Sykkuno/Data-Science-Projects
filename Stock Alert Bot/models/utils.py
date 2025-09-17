# SAmodel/utils.py
import re
import numpy as np
import pandas as pd

LABELS = {0: "down", 1: "up"}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\$[A-Za-z]+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s\.\,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def sentiment_to_score(pipeline_outputs):
    """
    Convert HF pipeline top-k outputs into a single numeric sentiment score.
    pipeline_outputs can be list-of-dicts (for a single string) or list-of-list for batch.
    We'll map label names to signed scores: positive -> +score, negative -> -score, neutral -> 0
    """
    def _single(preds):
        # preds is list of dicts like [{'label': 'LABEL_2', 'score':0.9}, ...] or {'label':'POS','score':0.8}
        # Accept both label formats e.g. 'POSITIVE' or 'LABEL_2'
        if isinstance(preds, dict):
            preds = [preds]
        s = 0.0
        for p in preds:
            lab = p.get("label", "").lower()
            score = float(p.get("score", 0.0))
            if "pos" in lab:
                s += score
            elif "neg" in lab:
                s -= score
            # neutral contributes 0
        return s

    # batch or single
    if isinstance(pipeline_outputs, list) and pipeline_outputs and isinstance(pipeline_outputs[0], list):
        return [ _single(item) for item in pipeline_outputs ]
    else:
        return _single(pipeline_outputs)

def aggregate_sentiments(sent_scores, window=3):
    """
    Given a list/Series of daily sentiment_scores (float), aggregates into a single feature:
    e.g., rolling mean of last `window` days, with NaN -> 0.
    """
    s = pd.Series(sent_scores).fillna(0.0)
    return s.rolling(window=window, min_periods=1).mean().iloc[-1]

def pct_change(series, periods=1):
    return series.pct_change(periods=periods)

def compute_label_next_day(df, price_col="close"):
    # Flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x]) for col in df.columns
        ]
    else:
        df.columns = [str(c) for c in df.columns]

    df.columns = [c.lower() for c in df.columns]

    # --- Auto-detect close column ---
    candidates = [c for c in df.columns if c.startswith("close")]
    if not candidates:
        raise KeyError(f"No close column found in {df.columns}")
    if price_col not in df.columns:
        # fallback to first detected close_*
        price_col = candidates[0]

    print(f"Using price column: {price_col}")

    # Ensure sorted
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # Next day's close
    df["next_close"] = df[price_col].shift(-1)

    df = df.dropna(subset=["next_close"])
    df["label"] = (df["next_close"].values > df[price_col].values).astype(int)

    return df


