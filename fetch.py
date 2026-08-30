"""일봉 데이터 수집. 미국은 Stooq, 한국은 FinanceDataReader를 씁니다.

두 소스 모두 API 키가 필요 없습니다.
어떤 소스를 쓰든 결과는 Open/High/Low/Close/Volume 컬럼과 날짜 인덱스로 통일합니다.
"""

import io
import time

import pandas as pd
import requests

TIMEOUT = 30
RETRIES = 3
NEEDED = ["Open", "High", "Low", "Close", "Volume"]


class FetchError(Exception):
    pass


def fetch(market: str, code: str) -> pd.DataFrame:
    """market: 'US' 또는 'KR'. code: 'AAPL' 또는 '005930'."""
    if market == "US":
        return _retry(_stooq_us, code)
    if market == "KR":
        return _retry(_fdr_kr, code)
    raise FetchError(f"알 수 없는 시장: {market}")


def _retry(fn, code):
    last = None
    for attempt in range(RETRIES):
        try:
            df = fn(code)
            _validate(df, code)
            return df
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)  # 1초, 2초로 늘려가며 재시도
    raise FetchError(f"{code}: {RETRIES}회 시도 실패 — {type(last).__name__}: {last}")


def _stooq_us(ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    text = r.text.strip()
    # 한도 초과나 없는 종목이면 CSV 대신 짧은 안내 문구가 옵니다
    if not text.startswith("Date,"):
        raise FetchError(f"CSV가 아닌 응답: {text[:80]!r}")
    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"]).set_index("Date")
    return df


def _fdr_kr(code: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    df = fdr.DataReader(code)
    df.index = pd.to_datetime(df.index)
    return df


def _validate(df: pd.DataFrame, code: str) -> None:
    if df is None or df.empty:
        raise FetchError(f"{code}: 빈 데이터")
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise FetchError(f"{code}: 컬럼 누락 {missing} (받은 컬럼: {list(df.columns)})")
    if df["Close"].tail(1).isna().all():
        raise FetchError(f"{code}: 마지막 종가가 비어 있음")
