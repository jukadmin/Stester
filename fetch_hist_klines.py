import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import requests
from pymongo import MongoClient, ASCENDING

# === Константы ===
BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL = "1m"
WINDOW_HOURS = 6
CHUNK_HOURS = 2
LIMIT = 500
MONGO_URI = "mongodb://localhost:27017"


# ---------- MongoDB utils ----------
def m_coll(symbol: str, db_name: str):
    client = MongoClient(MONGO_URI)
    coll = client[db_name][f"klines_1m_{symbol.lower()}"]
    if "timestamp_1" not in coll.index_information():
        coll.create_index([("timestamp", ASCENDING)], unique=True)
    return coll


# ---------- Binance utils ----------
def fetch_klines(symbol: str, start_ts_ms: int, end_ts_ms: int) -> List[List]:
    params = {
        "symbol": symbol.upper(),
        "interval": INTERVAL,
        "startTime": start_ts_ms,
        "endTime": end_ts_ms,
        "limit": LIMIT,
    }
    r = requests.get(BINANCE_FAPI_URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def normalize(raw: List[List]) -> List[Dict]:
    return [
        {
            "timestamp": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        }
        for k in raw
    ]


# ---------- Основная функция ----------
def get_klines(symbol: str, database_name: str):
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=WINDOW_HOURS)
    coll = m_coll(symbol, database_name)

    print(
        f"[{symbol}] Загружаем с {start_utc:%Y-%m-%d %H:%M:%S} UTC "
        f"по {now_utc:%Y-%m-%d %H:%M:%S} UTC, шагом {CHUNK_HOURS}ч"
    )

    total_inserted = 0
    step_ms = CHUNK_HOURS * 60 * 60 * 1000
    start_ts = int(start_utc.timestamp() * 1000)
    end_ts = int(now_utc.timestamp() * 1000)

    for chunk_start in range(start_ts, end_ts, step_ms):
        chunk_end = min(chunk_start + step_ms - 1, end_ts)
        dt_start = datetime.fromtimestamp(chunk_start / 1000, tz=timezone.utc)
        dt_end = datetime.fromtimestamp(chunk_end / 1000, tz=timezone.utc)
        print(f"  → От {dt_start:%H:%M} до {dt_end:%H:%M}")

        try:
            raw_klines = fetch_klines(symbol, chunk_start, chunk_end)
            docs = normalize(raw_klines)
            for d in docs:
                coll.update_one({"timestamp": d["timestamp"]}, {"$set": d}, upsert=True)
            print(f"    Загружено: {len(docs)} свечей")
            total_inserted += len(docs)
        except Exception as e:
            print(f"    ⚠ Ошибка: {e}")
            return False

        time.sleep(0.25)

    print(f"✅ Всего вставлено/обновлено: {total_inserted}")
    print("Готово.")
    return True


# ---------- Точка входа ----------
if __name__ == "__main__":
    get_klines("BTCUSDT", "cbxbot_db")
