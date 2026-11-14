"""시계열 저장소와 수집 상태.

저장 구조 (parquet):
    data/processed/{dataset}/{symbol}/{period}.parquet
    - dataset 예: binance_spot_1m, binance_um_1m, binance_um_funding, upbit_1m, kr_1m, kr_1d, us_1d, us_1m
    - period: 분봉은 YYYY-MM, 일봉 이상은 YYYY
    - ts 컬럼: 봉 시작 시각, UTC (tz-aware). 가격은 원본 그대로 저장한다.

수집 상태 (sqlite: data/state.sqlite):
    (dataset, symbol)마다 first_ts / last_ts / 상태 / 마지막 오류를 기록한다.
    수집이 중간에 끊겨도 다음 실행 때 이 기록부터 이어서 받는다.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
STATE_DB = ROOT / "data" / "state.sqlite"


def _period_key(ts: pd.Series, dataset: str) -> pd.Series:
    if dataset.endswith(("_1m", "_5m")):
        return ts.dt.strftime("%Y-%m")
    return ts.dt.strftime("%Y")


def write(dataset: str, symbol: str, df: pd.DataFrame, key: list[str] | None = None) -> int:
    """기존 파일과 병합해 저장 (key 기준 중복 제거, 나중 값 우선). 저장한 행 수를 반환.
    key 기본값은 ["ts"]. 여러 종목을 한 표에 담는 횡단면 데이터셋은 ["ts", "symbol"]처럼 지정한다."""
    key = key or ["ts"]
    if df.empty:
        return 0
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    folder = DATA_DIR / dataset / _safe(symbol)
    folder.mkdir(parents=True, exist_ok=True)
    for period, part in df.groupby(_period_key(df["ts"], dataset)):
        path = folder / f"{period}.parquet"
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part = part.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
        tmp = path.with_suffix(".tmp")
        part.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, path)
    set_state(dataset, symbol, first_ts=df["ts"].min(), last_ts=df["ts"].max())
    return len(df)


def read(dataset: str, symbol: str, start=None, end=None) -> pd.DataFrame:
    folder = DATA_DIR / dataset / _safe(symbol)
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if start is not None:
        df = df[df["ts"] >= _utc(start)]
    if end is not None:
        df = df[df["ts"] < _utc(end)]
    return df.sort_values("ts").reset_index(drop=True)


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def query(sql: str) -> pd.DataFrame:
    """여러 종목을 한 번에 다룰 때: duckdb SQL. 예) FROM 'data/processed/kr_1d/*/*.parquet'
    (주의: duckdb는 ts를 세션 시간대로 돌려주므로 UTC로 맞춰 둔다)"""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    return con.sql(sql).df()


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_").replace("^", "_")


# ── 수집 상태 ──
def _db() -> sqlite3.Connection:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(STATE_DB, timeout=60)  # 수집기 여러 개가 동시에 기록
    con.execute(
        """CREATE TABLE IF NOT EXISTS state (
            dataset TEXT, symbol TEXT, first_ts TEXT, last_ts TEXT,
            backfill_done INTEGER DEFAULT 0, status TEXT, error TEXT, updated_at TEXT,
            PRIMARY KEY (dataset, symbol))"""
    )
    return con


def get_state(dataset: str, symbol: str) -> dict:
    with _db() as con:
        row = con.execute(
            "SELECT first_ts, last_ts, backfill_done, status, error FROM state WHERE dataset=? AND symbol=?",
            (dataset, symbol),
        ).fetchone()
    if not row:
        return {"first_ts": None, "last_ts": None, "backfill_done": False, "status": None, "error": None}
    return {
        "first_ts": pd.Timestamp(row[0]) if row[0] else None,
        "last_ts": pd.Timestamp(row[1]) if row[1] else None,
        "backfill_done": bool(row[2]),
        "status": row[3],
        "error": row[4],
    }


def set_state(dataset: str, symbol: str, first_ts=None, last_ts=None, backfill_done=None,
              status: str | None = "ok", error: str | None = None) -> None:
    cur = get_state(dataset, symbol)
    if first_ts is not None and (cur["first_ts"] is None or first_ts < cur["first_ts"]):
        cur["first_ts"] = first_ts
    if last_ts is not None and (cur["last_ts"] is None or last_ts > cur["last_ts"]):
        cur["last_ts"] = last_ts
    if backfill_done is not None:
        cur["backfill_done"] = backfill_done
    with _db() as con:
        con.execute(
            "INSERT OR REPLACE INTO state VALUES (?,?,?,?,?,?,?,?)",
            (
                dataset, symbol,
                cur["first_ts"].isoformat() if cur["first_ts"] is not None else None,
                cur["last_ts"].isoformat() if cur["last_ts"] is not None else None,
                int(cur["backfill_done"]), status, error,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )


def summary() -> pd.DataFrame:
    with _db() as con:
        return pd.read_sql(
            "SELECT dataset, COUNT(*) symbols, MIN(first_ts) first_ts, MAX(last_ts) last_ts,"
            " SUM(backfill_done) backfilled, SUM(status='error') errors FROM state GROUP BY dataset",
            con,
        )


def save_universe(name: str, df: pd.DataFrame) -> None:
    """유니버스를 날짜별로 남긴다 (언제 어떤 종목을 골랐는지 기록 → 생존편향 추적)."""
    folder = DATA_DIR / "universe" / name
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / f"{datetime.now():%Y-%m-%d}.parquet", index=False)


def load_universe(name: str, max_age_days: int = 30) -> pd.DataFrame:
    """가장 최근 유니버스. max_age_days보다 오래됐으면 빈 값 → 호출 측에서 새로 만든다."""
    files = sorted((DATA_DIR / "universe" / name).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    age = datetime.now() - datetime.strptime(files[-1].stem, "%Y-%m-%d")
    return pd.read_parquet(files[-1]) if age.days <= max_age_days else pd.DataFrame()
