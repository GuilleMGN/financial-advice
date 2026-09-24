"""
Send a push notification via Firebase Cloud Messaging (FCM), to a topic
rather than a specific device. Any device that later subscribes to this
topic (the phone app, once built) receives it.

Setup:
    pip install firebase-admin

Usage:
    python send_notification.py --key path/to/serviceAccountKey.json ^
        --title "ZUE.TO BUY signal" --body "RSI 27.4 on the 4H close"

    (On Mac/Linux use \\ instead of ^ for line continuation, or just put
    it all on one line.)
"""

import argparse

import firebase_admin
from firebase_admin import credentials, messaging

TOPIC = "trading-alerts"


def send(key_path: str, title: str, body: str, topic: str = TOPIC):
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)

    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        topic=topic,
    )
    response = messaging.send(message)
    print(f"Sent. Message ID: {response}")


def main():
    parser = argparse.ArgumentParser(description="Send a test FCM push notification")
    parser.add_argument("--key", required=True, help="path to Firebase service account JSON")
    parser.add_argument("--title", default="Test notification", help="notification title")
    parser.add_argument("--body", default="This is a test from the RSI alert backend.",
                         help="notification body text")
    parser.add_argument("--topic", default=TOPIC, help="FCM topic to send to")
    args = parser.parse_args()

    send(args.key, args.title, args.body, args.topic)


if __name__ == "__main__":
    main()
