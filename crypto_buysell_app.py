#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crypto_buysell_app.py

Version 1.14 – Enhanced backtest UI with metrics and full SMC and Order Flow sections.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import io

import yaml
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF

from joblib import Memory
from pytrends.request import TrendReq

# SMC tools
from smc_tools import (
    detect_swing_high_low,
    detect_liquidity_sweeps,
    compute_volume_profile,
    detect_supply_demand_zones,
    load_orderbook_csv,
    compute_order_flow
)

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
#  DATA FETCHING & INDICATORS
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


def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # [Indicator code unchanged]
    # RSI, MACD, SMA, OBV, EMA, Bollinger, ATR, CMF
    # ... implementation omitted for brevity ...
    return df


def compute_macro_strength(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    # ... unchanged ...
    return df


def fetch_google_trends(df: pd.DataFrame, symbol: str, start: str, end: str, cfg: dict) -> pd.DataFrame:
    # ... unchanged ...
    return df


def fetch_fear_greed(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    resp = requests.get("https://api.alternative.me/fng/").json()
    val  = int(resp["data"][0]["value"])
    df["fng_signal"] = int(val < cfg["fng_threshold"])
    df["fng"]        = val
    return df


def score_and_signal(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # ... unchanged ...
    return df


def backtest(df: pd.DataFrame, cfg: dict) -> dict:
    # ... unchanged ...
    return {}

# ──────────────────────────────────────────────────────────────────────────────
#  STREAMLIT DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
def run_dashboard():
    st.set_page_config(page_title="Crypto Buy/Sell Signals", layout="wide")
    st.title("🚀 Crypto Buy/Sell Signals Dashboard")

    cfg = load_config(Path("config.yaml"))

    # Sidebar
    symbol     = st.sidebar.text_input("Ticker", "ETH-USD")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime(cfg.get("start_date","2022-01-01")))
    score_th   = st.sidebar.slider("Score Threshold", 1, 20, cfg["score_thresh"])
    show_smc   = st.sidebar.checkbox("Enable SMC Tools", False)

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
        stats = backtest(df, cfg)

        # Prepare backtest metrics
        buys = df[df.buy_signal==1].copy()
        horizon = cfg["backtest_horizon_days"]
        buys["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
        total_trades = len(buys)
        avg_return = stats.get("avg_return", 0)
        win_rate   = stats.get("win_rate", 0)
        min_return = buys["future_return"].min() if total_trades>0 else 0
        max_return = buys["future_return"].max() if total_trades>0 else 0

        # Tabs
        tabs = st.tabs([
            "Summary","Price & Signals","Signal Details",
            *( ["SMC Swings","SMC Sweeps","Volume Profile","Supply/Demand"] if show_smc else [] ),
            *( ["Order Flow"] if show_smc else [] )
        ])
        fig_dict = {}

        # Summary Tab
        with tabs[0]:
            st.subheader("Backtest Summary")
            cols = st.columns(6)
            cols[0].metric("Start Date", start)
            cols[1].metric("End Date", end)
            cols[2].metric("Trades", total_trades)
            cols[3].metric("Avg Return", f"{avg_return:.2%}")
            cols[4].metric("Win Rate", f"{win_rate:.2%}")
            cols[5].metric("Best Return", f"{max_return:.2%}")
            st.metric("Worst Return", f"{min_return:.2%}")

        # Price & Signals Tab
        with tabs[1]:
            st.subheader("Price & Score")
            fig, ax = plt.subplots()
            ax.plot(df["close"], label="Price")
            ax2 = ax.twinx()
            ax2.plot(df["score"], label="Score", alpha=0.7)
            ax.legend(loc="upper left"); ax2.legend(loc="upper right")
            st.pyplot(fig)
            fig_dict["Price & Score"] = fig

        # Signal Details
        with tabs[2]:
            st.subheader("Buy Signal Details")
            st.dataframe(buys[["close","score","trend_signal","fng_signal","future_return"]])

        # SMC and Order Flow Tabs
        idx = 3
        if show_smc:
            # Swing Highs/Lows
            with tabs[idx]:
                smc_df = detect_swing_high_low(df)
                st.subheader("SMC: Swing Highs & Lows")
                st.dataframe(smc_df[smc_df.swing_high | smc_df.swing_low][["high","low","swing_high","swing_low"]])
            idx += 1

            # Liquidity Sweeps
            with tabs[idx]:
                smc_df = detect_liquidity_sweeps(smc_df)
                st.subheader("SMC: Liquidity Sweeps")
                st.dataframe(smc_df[smc_df.sweep_high | smc_df.sweep_low][["high","low","sweep_high","sweep_low"]])
            idx += 1

            # Volume Profile
            with tabs[idx]:
                vp = compute_volume_profile(df)
                st.subheader("SMC: Volume Profile")
                vp_plot = vp.copy()
                vp_plot["price_mid"] = vp_plot["price_bin"].apply(lambda x: (x.left + x.right)/2)
                fig2, ax2 = plt.subplots()
                ax2.bar(vp_plot["price_mid"], vp_plot["volume"], width=(vp_plot["price_bin"].iloc[0].length))
                ax2.set_xlabel("Price"); ax2.set_ylabel("Volume")
                st.pyplot(fig2)
                fig_dict["Volume Profile"] = fig2
            idx += 1

            # Supply/Demand Zones
            with tabs[idx]:
                zones = detect_supply_demand_zones(vp)
                st.subheader("SMC: Supply/Demand Zones")
                st.dataframe(zones)
            idx += 1

            # Order Flow
            with tabs[idx]:
                st.subheader("SMC: Order Flow Analysis")
                orderbook_file = st.file_uploader("Upload order-book CSV", type="csv")
                if orderbook_file:
                    ob_df = load_orderbook_csv(orderbook_file)
                    of_df = compute_order_flow(ob_df)
                    fig3, ax3 = plt.subplots()
                    ax3.plot(of_df["cumulative_delta"], label="Cumulative Delta")
                    ax3.set_title("Cumulative Delta")
                    st.pyplot(fig3)
                    fig_dict["Cumulative Delta"] = fig3

                    fig4, ax4 = plt.subplots()
                    ax4.plot(of_df["imbalance"], label="Imbalance")
                    ax4.set_title("Imbalance")
                    st.pyplot(fig4)
                    fig_dict["Imbalance"] = fig4

        # PDF Report
        st.markdown("---")
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"Crypto Report: {symbol}", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, f"Period: {start} to {end}", ln=True)
            pdf.ln(5)
            for title, fig in fig_dict.items():
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, title, ln=True)
                pdf.image(buf, x=10, y=30, w=pdf.epw - 20)
                buf.close()
            pdf_data = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name="crypto_report.pdf",
                mime="application/pdf"
            )

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
        print("  streamlit run crypto_buysell_app.py -- --dashboard")

if __name__ == "__main__":
    main()
