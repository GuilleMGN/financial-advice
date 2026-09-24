"""
4H RSI Signal Prototype
------------------------
Goal: validate the trading-signal logic BEFORE we build any app around it.

Logic being tested:
  - Pull 1H candles for a stock/ETF (Yahoo Finance ticker, e.g. "ZUE.TO" for
    a TSX-listed hedged ETF).
  - Resample into plain, midnight-aligned 4H candles (no attempt to line up
    with the 9:30am market open - simpler). In practice this gives a short
    first candle each day (9:30-12:00) and one clean 12:00-16:00 candle.
  - Compute RSI(14) using Wilder's smoothing (the standard RSI method).
  - The MOMENT a 4H candle closes, check its own RSI value:
        RSI < 30  -> BUY signal
        RSI > 70  -> SELL signal

Setup (run this on your own machine, not in this sandbox - it needs
internet access to reach Yahoo Finance):

    pip install yfinance pandas numpy

Usage:

    python rsi_prototype.py --ticker ZUE.TO --days 60
    python rsi_prototype.py --watchlist ZUE.TO,XSP.TO,HXS.TO --days 60

Notes:
  - Yahoo Finance only keeps 1H intraday history for the last ~730 days,
    and some tickers have thinner intraday history than others.
  - This script only PRINTS what it finds. No notifications yet - that's
    a later step once we're happy with the logic itself.
"""

import argparse
import sys

import numpy as np
import pandas as pd


def fetch_hourly(ticker: str, days: int) -> pd.DataFrame:
    """Pull 1H OHLC data from Yahoo Finance for the given ticker."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Missing dependency. Run: pip install yfinance pandas numpy")

    period = f"{min(days, 729)}d"  # yfinance caps 1h data at ~730 days
    df = yf.download(ticker, period=period, interval="1h", progress=False)

    if df.empty:
        sys.exit(f"No data returned for '{ticker}'. Check the ticker symbol "
                  f"(TSX tickers need the '.TO' suffix, e.g. 'ZUE.TO').")

    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")

    return df[["Open", "High", "Low", "Close", "Volume"]]


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1H candles into plain 4H candles, midnight-aligned
    (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 ET). No attempt to line up
    with the 9:30am market open - simpler, and since market data only
    exists 9:30-16:00 anyway, this naturally produces a short first candle
    each day (9:30-12:00) followed by one clean 12:00-16:00 candle.
    """
    out = df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    # Drop empty bins (weekends, overnight gaps, holidays)
    out = out.dropna(subset=["Open", "Close"])
    return out


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI - the standard RSI calculation used by most platforms."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 (straight uptrend), RSI should be 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def find_signals(df: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> pd.DataFrame:
    """
    For each candle, check its OWN closing RSI value the moment it closes.
    This is the trigger moment for a notification: candle closes -> RSI
    checked -> signal fires if below oversold or above overbought.
    """
    df = df.copy()

    def classify(rsi):
        if pd.isna(rsi):
            return None
        if rsi < oversold:
            return "BUY"
        if rsi > overbought:
            return "SELL"
        return None

    df["signal"] = df["rsi"].apply(classify)
    return df


def analyze_ticker(ticker: str, days: int, rsi_period: int, oversold: float, overbought: float):
    """Run the full pipeline for one ticker, return (four_h_df, latest_row, all_signals)."""
    hourly = fetch_hourly(ticker, days)
    four_h = resample_to_4h(hourly)
    four_h["rsi"] = compute_rsi(four_h["Close"], period=rsi_period)
    four_h = find_signals(four_h, oversold, overbought)
    latest = four_h.iloc[-1]
    signals = four_h[four_h["signal"].notna()]
    return four_h, latest, signals


def main():
    parser = argparse.ArgumentParser(description="Prototype 4H RSI signal logic")
    parser.add_argument("--ticker", help="single ticker, e.g. ZUE.TO")
    parser.add_argument("--watchlist", help="comma-separated tickers, e.g. ZUE.TO,XSP.TO,HXS.TO")
    parser.add_argument("--days", type=int, default=60, help="lookback window in days")
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--oversold", type=float, default=30)
    parser.add_argument("--overbought", type=float, default=70)
    args = parser.parse_args()

    if not args.ticker and not args.watchlist:
        sys.exit("Provide --ticker ZUE.TO  OR  --watchlist ZUE.TO,XSP.TO,HXS.TO")

    tickers = args.watchlist.split(",") if args.watchlist else [args.ticker]
    tickers = [t.strip() for t in tickers if t.strip()]

    print(f"Watchlist: {', '.join(tickers)}\n")

    summary_rows = []
    all_signal_tables = {}

    for ticker in tickers:
        print(f"--- {ticker} ---")
        try:
            four_h, latest, signals = analyze_ticker(
                ticker, args.days, args.rsi_period, args.oversold, args.overbought
            )
        except SystemExit as e:
            print(f"  Skipped: {e}")
            continue

        print(f"  {len(four_h)} 4H candles built. Latest close: {latest['Close']:.2f}, "
              f"latest RSI: {latest['rsi']:.2f}, current signal: {latest['signal']}")
        print(f"  Signals in window: {len(signals)}")

        summary_rows.append({
            "ticker": ticker,
            "latest_close": round(latest["Close"], 2),
            "latest_rsi": round(latest["rsi"], 2),
            "current_signal": latest["signal"],
            "signal_count": len(signals),
        })
        all_signal_tables[ticker] = signals

    print("\n=== Watchlist summary (as of latest closed 4H candle) ===")
    summary_df = pd.DataFrame(summary_rows).set_index("ticker")
    print(summary_df.to_string())

    active = summary_df[summary_df["current_signal"].notna()]
    if not active.empty:
        print("\n*** ACTIVE SIGNALS RIGHT NOW ***")
        print(active.to_string())
    else:
        print("\nNo active signals on the most recent candle for any watchlist ticker.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
