import os
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}


def fetch_twitter_stats(username, client_id):
    url = f"https://api.twitter.com/2/users/by/username/{username}?user.fields=public_metrics"
    response = requests.get(url, headers=headers).json()
    stats = response['data']['public_metrics']

    df = pd.DataFrame([{
        "client_id": client_id,
        "platform": "Twitter",
        "date": datetime.now().date(),
        "followers": stats['followers_count'],
        "impressions": stats['tweet_count'],  # demo
        "engagement_rate": stats['following_count'] / max(stats['followers_count'], 1)
    }])
    df.to_sql("daily_stats", engine, if_exists="append", index=False)
    print("Twitter data inserted!")


fetch_twitter_stats("TwitterDev", "client_1")
