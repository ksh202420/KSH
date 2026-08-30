"""일봉 데이터 수집.

시장별로 소스를 여러 개 순서대로 시도합니다. 앞의 것이 실패하면 다음으로 넘어가고,
어떤 소스로 성공했는지 로그에 남깁니다. 어느 소스가 살아 있는지 알아야
나중에 순서를 바꿀 수 있기 때문입니다.

  미국: yfinance → Stooq
  한국: FinanceDataReader → pykrx

모두 API 키가 필요 없습니다.
결과는 소스와 무관하게 Open/High/Low/Close/Volume 컬럼과 날짜 인덱스로 통일합니다.
"""

import io
import time

import pandas as pd
import requests

TIMEOUT = 30
ATTEMPTS = 2          # 소스 하나당 시도 횟수
NEEDED = ["Open", "High", "Low", "Close", "Volume"]


class FetchError(Exception):
    pass


# ── 소스별 수집 함수 ──────────────────────────────────────────

def _yfinance_us(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):        # 버전에 따라 다층 컬럼이 옵니다
        df.columns = df.columns.get_level_values(0)
    return df


def _stooq_us(ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    text = r.text.strip()
    # 한도 초과·차단이면 CSV 대신 HTML 페이지가 돌아옵니다
    if not text.startswith("Date,"):
        raise FetchError(f"CSV가 아닌 응답: {text[:60]!r}")
    return pd.read_csv(io.StringIO(text), parse_dates=["Date"]).set_index("Date")


def _fdr_kr(code: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    df = fdr.DataReader(code)
    df.index = pd.to_datetime(df.index)
    return df


def _pykrx_kr(code: str) -> pd.DataFrame:
    from datetime import date, timedelta

    from pykrx import stock

    end = date.today()
    start = end - timedelta(days=365 * 6)          # 200일선·12개월 모멘텀에 넉넉하게
    df = stock.get_market_ohlcv(start.strftime("%Y%m%d"),
                                end.strftime("%Y%m%d"), code)
    df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low",
                            "종가": "Close", "거래량": "Volume"})
    df.index = pd.to_datetime(df.index)
    return df


SOURCES = {
    "US": [("yfinance", _yfinance_us), ("stooq", _stooq_us)],
    "KR": [("FinanceDataReader", _fdr_kr), ("pykrx", _pykrx_kr)],
}


# ── 공개 함수 ─────────────────────────────────────────────────

def fetch(market: str, code: str) -> pd.DataFrame:
    """market: 'US' 또는 'KR'. code: 'AAPL' 또는 '005930'."""
    chain = SOURCES.get(market)
    if not chain:
        raise FetchError(f"알 수 없는 시장: {market}")

    problems = []
    for name, fn in chain:
        for attempt in range(ATTEMPTS):
            try:
                df = fn(code)
                _validate(df, code)
                if name != chain[0][0]:
                    print(f"       ↳ {code}: 주 소스 실패 — 백업 사용 ({name})")
                return df[NEEDED].copy()
            except Exception as e:  # noqa: BLE001
                note = f"{name}: {type(e).__name__}: {e}"
                if note not in problems:      # 재시도로 같은 오류가 겹치는 것을 막습니다
                    problems.append(note)
                if attempt < ATTEMPTS - 1:
                    time.sleep(1.5)

    # 모든 소스가 실패한 경우에만 여기까지 옵니다
    raise FetchError(f"{code}: 모든 소스 실패 — " + " | ".join(_short(p) for p in problems))


def _validate(df: pd.DataFrame, code: str) -> None:
    if df is None or df.empty:
        raise FetchError("빈 데이터")
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise FetchError(f"컬럼 누락 {missing} (받은 컬럼: {list(df.columns)})")
    if df["Close"].tail(1).isna().all():
        raise FetchError("마지막 종가가 비어 있음")
    if len(df) < 30:
        raise FetchError(f"데이터가 {len(df)}행뿐 — 지표 계산 불가")


def _short(msg: str, n: int = 90) -> str:
    """로그가 읽히도록 긴 오류 메시지를 잘라냅니다."""
    msg = " ".join(msg.split())
    return msg if len(msg) <= n else msg[:n] + "…"
