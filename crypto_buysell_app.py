#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crypto_buysell_app.py

Version 1.11 – Full interactive Streamlit UI added.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
import yfinance as yf
import requests

from joblib import Memory
from pytrends.request import TrendReq

# ──────────────────────────────────────────────────────────────────────────────
#  SETTINGS & CACHING
# ──────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".cache"
memory = Memory(location=CACHE_DIR, verbose=0)

def load_config(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ──────────────────────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)
log = logging.getLogger("crypto_buysell")

# ──────────────────────────────────────────────────────────────────────────────
#  DATA FETCHING (cached)
# ──────────────────────────────────────────────────────────────────────────────
@memory.cache
def fetch_price(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=False)
    df = df[["Open","High","Low","Close","Volume"]]
    df.columns = ["open","high","low","close","volume"]
    return df

@memory.cache
def fetch_macro(macros: dict, start: str, end: str) -> pd.DataFrame:
    data = yf.download(list(macros.values()), start=start, end=end)[["Close"]]
    if isinstance(data.columns, pd.MultiIndex):
        df = data["Close"].copy()
    else:
        df = data.copy()
    df.rename(columns={v:k for k,v in macros.items()}, inplace=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
#  INDICATORS & SIGNALS
# ──────────────────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # RSI
    d = df["close"].diff()
    g, l = d.clip(lower=0), -d.clip(upper=0)
    avg_g = g.rolling(cfg["rsi_len"]).mean()
    avg_l = l.rolling(cfg["rsi_len"]).mean()
    df["rsi"] = 100 - (100 / (1 + avg_g/avg_l))
    # MACD hist
    ef = df["close"].ewm(span=12, adjust=False).mean()
    es = df["close"].ewm(span=26, adjust=False).mean()
    macd = ef - es
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    # Volume SMA
    df["vol_sma"]   = df["volume"].rolling(cfg["vol_sma"]).mean()
    # OBV
    df["obv"]       = (np.sign(df["close"].diff().fillna(0)) * df["volume"]).cumsum()
    df["obv_sma"]   = df["obv"].rolling(cfg["vol_sma"]).mean()
    # EMA cross
    df["ema_short"] = df["close"].ewm(span=cfg["ema_short"], adjust=False).mean()
    df["ema_long"]  = df["close"].ewm(span=cfg["ema_long"], adjust=False).mean()
    # Bollinger lower
    mid = df["close"].rolling(cfg["bb_window"]).mean()
    sd  = df["close"].rolling(cfg["bb_window"]).std()
    df["bb_lower"]  = mid - 2*sd
    # ATR breakout
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"]             = tr.rolling(cfg["atr_len"]).mean()
    df["atr_breakout"]    = (df["close"] > df["close"].shift(1) + df["atr"]).astype(int)
    # CMF
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf_mult = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    mf_mult.fillna(0, inplace=True)
    mf = mf_mult * df["volume"]
    df["cmf"]            = mf.rolling(cfg["vol_sma"]).sum() / df["volume"].rolling(cfg["vol_sma"]).sum()
    df["cmf_signal"]     = (df["cmf"] > 0).astype(int)
    return df

def compute_macro_strength(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([df["close"], macro], axis=1).dropna()
    rets = data.pct_change().dropna()
    zs   = (rets - rets.mean()) / rets.std()
    zs["DXY"] *= -1; zs["VIX"] *= -1
    df = df.join(zs.mean(axis=1).rename("macro_str"))
    return df

def fetch_google_trends(df: pd.DataFrame, symbol: str, start: str, end: str, cfg: dict) -> pd.DataFrame:
    kw = f"{symbol.split('-')[0]} cryptocurrency"
    py = TrendReq(hl='en-US', tz=360)
    py.build_payload([kw], timeframe=f"{start} {end}")
    tr = py.interest_over_time().drop(columns=["isPartial"])
    tr.rename(columns={kw: "trend_interest"}, inplace=True)
    df = df.join(tr, how="left").ffill()
    thresh = df["trend_interest"].rolling(cfg["trend_lookback"]).quantile(0.9)
    df["trend_signal"] = (df["trend_interest"] > thresh).astype(int)
    return df

def fetch_fear_greed(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    resp = requests.get("https://api.alternative.me/fng/").json()
    val  = int(resp["data"][0]["value"])
    df["fng_signal"] = (val < cfg["fng_threshold"]).astype(int)
    df["fng"]        = val
    return df

def score_and_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df["score"] = (
          (df["rsi"]          < 30).astype(int)
        + (df["macd_hist"]    > 0).astype(int)
        + (df["volume"]       > df["vol_sma"]).astype(int)
        + (df["obv"]          > df["obv_sma"]).astype(int)
        + (df["ema_short"]    > df["ema_long"]).astype(int)
        + (df["close"]        < df["bb_lower"]).astype(int)
        + df["atr_breakout"]
        + df["cmf_signal"]
        + (df["macro_str"]    > 0).astype(int)
        + df["trend_signal"]
        + df["fng_signal"]
    )
    df["buy_signal"] = (df["score"] >= cfg["score_thresh"]).astype(int)
    return df

def backtest(df: pd.DataFrame, cfg: dict) -> dict:
    n = cfg["backtest_horizon_days"]
    df["future_return"] = df["close"].shift(-n) / df["close"] - 1
    buys = df[df["buy_signal"]==1]
    return {
        "avg_return": buys["future_return"].mean(),
        "win_rate":   (buys["future_return"]>0).mean(),
        "n_trades":   len(buys)
    }

# ──────────────────────────────────────────────────────────────────────────────
#  STREAMLIT DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
def run_dashboard():
    import streamlit as st
    st.title("🚀 Crypto Buy/Sell Signals Dashboard")

    # Load config
    cfg = load_config(Path("config.yaml"))

    # Sidebar inputs
    symbol     = st.sidebar.text_input("Ticker", "ETH-USD")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime(cfg.get("start_date","2022-01-01")))
    score_th   = st.sidebar.slider("Score Threshold", 1, 20, cfg["score_thresh"])
    if st.sidebar.button("Run Analysis"):
        start = str(start_date)
        end   = datetime.now().strftime("%Y-%m-%d")

        # Pipeline
        df    = fetch_price(symbol, start, end)
        macro = fetch_macro(cfg["macros"], start, end)
        df    = compute_indicators(df.copy(), cfg)
        df    = compute_macro_strength(df, macro)
        df    = fetch_google_trends(df, symbol, start, end, cfg)
        df    = fetch_fear_greed(df, cfg)
        df    = score_and_signal(df, cfg)

        # Backtest
        stats = backtest(df, cfg)
        st.subheader("Backtest Summary")
        st.json(stats)

        # Charts
        st.subheader("Price & Buy Signals")
        chart = df[["close","score"]].rename(columns={"close":"Price","score":"Score"})
        st.line_chart(chart)

        st.subheader("Signal Details")
        st.dataframe(df[df.buy_signal==1][["close","score","trend_signal","fng_signal"]])

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dashboard", action="store_true", help="Run Streamlit dashboard")
    args = p.parse_args()

    if args.dashboard:
        run_dashboard()
    else:
        print("This script is intended to be run as a Streamlit app:")
        print("  streamlit run crypto_buysell_app.py")

if __name__ == "__main__":
    main()
