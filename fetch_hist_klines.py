from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import List, Dict, Tuple
import pandas as pd
import requests
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.collection import Collection


"""
Скачивает минутные фьючерс‑свечи с Binance, складывает в MongoDB,
при полной загрузке запрашиваемого диапазона экспортирует CSV.

✓  Проверяет, нет ли уже всех данных в Mongo → если есть, сразу CSV
✓  Для каждого 2‑часового чанка проверяет наличие и пропускает заполненные
✓  Чтение из Mongo в экспорт — потоково (batch_size), без list(find(...))
✓  Завершает сессию MongoClient (client.close()) во избежание утечек
✓  Совместимо с Python 3.13+
"""


# === Настройки ===
START_DATETIME = "2025-01-01 00:00:00"
END_DATETIME = "2025-07-05 00:00:00"
INTERVAL = "1m"
CHUNK_HOURS = 24
LIMIT = 1500
MONGO_URI = "mongodb://localhost:27017"

BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/klines"


# --------------------------------------------------------------------------- #
#                               MongoDB helpers                               #
# --------------------------------------------------------------------------- #


def mongo_client_and_coll(symbol: str, db_name: str) -> Tuple[MongoClient, Collection]:
    client = MongoClient(MONGO_URI)
    coll = client[db_name][f"hist_klines_1m_{symbol.lower()}"]
    if "timestamp_1" not in coll.index_information():
        coll.create_index([("timestamp", ASCENDING)], unique=True)
    return client, coll


def has_full_range(coll, start_ts: int, end_ts: int) -> bool:
    """
    Возвращает True, если в коллекции есть ровно
    (end_ts - start_ts) // 60_000 документов с timestamp ∈ [start, end)
    """
    expected = (end_ts - start_ts) // 60_000
    actual = coll.count_documents({"timestamp": {"$gte": start_ts, "$lt": end_ts}})
    return actual == expected


# --------------------------------------------------------------------------- #
#                               Binance helpers                               #
# --------------------------------------------------------------------------- #


def fetch_klines(symbol: str, start_ts_ms: int, end_ts_ms: int) -> List[List]:
    resp = requests.get(
        BINANCE_FAPI_URL,
        params={
            "symbol": symbol.upper(),
            "interval": INTERVAL,
            "startTime": start_ts_ms,
            "endTime": end_ts_ms,
            "limit": LIMIT,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


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


# --------------------------------------------------------------------------- #
#                               CSV export                                    #
# --------------------------------------------------------------------------- #


def export_to_csv(coll, start_ts: int, end_ts: int, output_file: str) -> None:
    """
    Потоково читает данные из Mongo и сохраняет в CSV.
    """
    cursor = coll.find(
        {"timestamp": {"$gte": start_ts, "$lt": end_ts}},
        {"_id": 0, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        batch_size=1_000,
       ) # no_cursor_timeout=True

    rows: list[dict] = []
    for doc in cursor:
        rows.append(doc)
    cursor.close()

    if not rows:
        print("⚠ CSV не создан: нет данных.")
        return

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(output_file, index=False)
    print(f"📄 CSV сохранён: {output_file}")


# --------------------------------------------------------------------------- #
#                               Main logic                                    #
# --------------------------------------------------------------------------- #


def get_klines(symbol: str, database_name: str) -> bool:
    start_dt = datetime.strptime(START_DATETIME, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END_DATETIME, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    client, coll = mongo_client_and_coll(symbol, database_name)

    try:
        # ---------- шаг 1. предварительная проверка ------------------------
        if has_full_range(coll, start_ts, end_ts):
            print("✅ Данные уже в Mongo. Пропускаем скачивание.")
            csv_name = f"{symbol.upper()}_{INTERVAL}_{START_DATETIME[:10]}.csv"
            export_to_csv(coll, start_ts, end_ts, csv_name)
            return True

        # ---------- шаг 2. скачивание чанками ------------------------------
        print(
            f"[{symbol}] Загрузка с {start_dt:%Y-%m-%d %H:%M:%S} UTC "
            f"по {end_dt:%Y-%m-%d %H:%M:%S} UTC, шаг {CHUNK_HOURS}ч"
        )

        step_ms = CHUNK_HOURS * 60 * 60 * 1000
        total_upserts = 0

        for chunk_start in range(start_ts, end_ts, step_ms):
            chunk_end_exc = min(chunk_start + step_ms, end_ts)  # [start, end)
            dt_start = datetime.fromtimestamp(chunk_start / 1000, tz=timezone.utc)
            dt_end = datetime.fromtimestamp((chunk_end_exc - 1) / 1000, tz=timezone.utc)
            print(f"  → {dt_start:%Y-%m-%d %H:%M}–{dt_end:%Y-%m-%d %H:%M}", end="")

            if has_full_range(coll, chunk_start, chunk_end_exc):
                print(" (уже есть, пропускаем)")
                continue

            try:
                raw = fetch_klines(symbol, chunk_start, chunk_end_exc - 1)
            except Exception as e:
                print(f"\n    ⚠ Ошибка запроса: {e}")
                return False

            docs = normalize(raw)
            if not docs:
                print(" — пусто")
                continue

            # bulk_write для уменьшения round‑trip
            requests_bulk = [
                UpdateOne({"timestamp": d["timestamp"]}, {"$set": d}, upsert=True) for d in docs
            ]
            result = coll.bulk_write(requests_bulk, ordered=False)
            upserted = result.upserted_count + result.modified_count
            total_upserts += upserted
            print(f" — upsert {upserted}")

            time.sleep(0.25)  # чуть быстрее

        # ---------- шаг 3. финальная проверка и CSV ------------------------
        print(f"✅ Всего вставлено/обновлено: {total_upserts}")
        if has_full_range(coll, start_ts, end_ts):
            csv_name = f"{symbol.upper()}_{INTERVAL}_{START_DATETIME[:10]}_{END_DATETIME[:10]}.csv"
            export_to_csv(coll, start_ts, end_ts, csv_name)
        else:
            print("❌ CSV не создан: диапазон заполнен не полностью.")
        return True
    finally:
        client.close()


# --------------------------------------------------------------------------- #
#                               Entry point                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    get_klines("DOTUSDT", "cbxbot_db")
