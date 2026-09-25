"""
check_signals.py
-----------------
The core backend job. For every ticker in the Firestore "watchlist"
collection: run the 4H RSI signal check, and if the most recently closed
candle has a BUY/SELL signal, send a push notification via FCM (topic
"trading-alerts") - but only ONCE per candle, even if this script runs
multiple times before the next candle closes. Tracked in Firestore via a
"last_notified_candle" field on each watchlist document.

Requires rsi_prototype.py in the same folder (reuses its functions).

Usage:
    python check_signals.py --key path/to/serviceAccountKey.json

Intended to run on a schedule (step 7: GitHub Actions), not by hand forever.
"""

from __future__ import annotations

import argparse

import firebase_admin
from firebase_admin import credentials, firestore, messaging

from rsi_prototype import fetch_hourly, resample_to_4h, compute_rsi, find_signals

TOPIC = "trading-alerts"
LOOKBACK_DAYS = 30  # enough history for RSI(14) to warm up properly


def init_firebase(key_path: str):
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def get_watchlist(db) -> list:
    docs = db.collection("watchlist").stream()
    return [doc.id for doc in docs]


def send_signal_notification(ticker: str, signal: str, price: float, rsi: float):
    title = f"{ticker} {signal} signal"
    body = f"4H RSI {rsi:.1f} at close, price {price:.2f}"
    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        topic=TOPIC,
    )
    response = messaging.send(message)
    print(f"  Notification sent for {ticker}: {response}")


def check_ticker(db, ticker: str, oversold: float, overbought: float):
    try:
        hourly = fetch_hourly(ticker, LOOKBACK_DAYS)
        four_h = resample_to_4h(hourly)
        four_h["rsi"] = compute_rsi(four_h["Close"])
        four_h = find_signals(four_h, oversold, overbought)
    except SystemExit as e:
        print(f"  Skipped {ticker}: {e}")
        return

    latest = four_h.iloc[-1]
    latest_ts = four_h.index[-1].isoformat()

    if latest["signal"] is None:
        print(f"  {ticker}: no signal (RSI {latest['rsi']:.1f})")
        return

    doc_ref = db.collection("watchlist").document(ticker)
    doc = doc_ref.get()
    existing = doc.to_dict() if doc.exists else {}
    last_notified_ts = existing.get("last_notified_candle")

    if last_notified_ts == latest_ts:
        print(f"  {ticker}: {latest['signal']} signal already notified for this candle, skipping")
        return

    print(f"  {ticker}: NEW {latest['signal']} signal (RSI {latest['rsi']:.1f})")
    send_signal_notification(ticker, latest["signal"], latest["Close"], latest["rsi"])
    doc_ref.set(
        {"last_notified_candle": latest_ts, "last_signal": latest["signal"]},
        merge=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Check watchlist for RSI signals and notify")
    parser.add_argument("--key", required=True, help="path to Firebase service account JSON")
    parser.add_argument("--oversold", type=float, default=30)
    parser.add_argument("--overbought", type=float, default=70)
    args = parser.parse_args()

    db = init_firebase(args.key)
    tickers = get_watchlist(db)

    if not tickers:
        print("Watchlist is empty. Add tickers from the app first.")
        return

    print(f"Checking {len(tickers)} ticker(s): {', '.join(tickers)}\n")
    for ticker in tickers:
        print(f"--- {ticker} ---")
        check_ticker(db, ticker, args.oversold, args.overbought)


if __name__ == "__main__":
    main()
