"""기술적 지표 계산. 외부 네트워크 없이 순수 계산만 하므로 단독으로 테스트할 수 있습니다."""

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()


def _wilder_avg(s: pd.Series, n: int) -> pd.Series:
    """Wilder 평활 평균.

    표준 정의는 '첫 n개의 단순평균으로 시작한 뒤 (이전값*(n-1) + 신규)/n'입니다.
    pandas ewm에 그냥 넣으면 첫 값 하나로 시드해서 초기 오차가 20포인트까지 벌어집니다.
    그래서 n번째 위치에 단순평균을 직접 심어주고 거기서부터 평활합니다.
    """
    s = s.astype(float)
    valid = s.dropna()
    if len(valid) < n:
        return pd.Series(np.nan, index=s.index, dtype=float)
    seeded = valid.copy()
    seeded.iloc[: n - 1] = np.nan
    seeded.iloc[n - 1] = valid.iloc[:n].mean()
    smoothed = seeded.ewm(alpha=1 / n, adjust=False).mean()
    return smoothed.reindex(s.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    avg_gain = _wilder_avg(delta.clip(lower=0), n)
    avg_loss = _wilder_avg(-delta.clip(upper=0), n)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    # 하락이 전혀 없으면 avg_loss = 0 → RS 무한대 → RSI 100
    out[(avg_loss == 0) & avg_gain.notna()] = 100.0
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR. True Range의 Wilder 평활 평균."""
    return _wilder_avg(true_range(df), n)


def momentum_12_1(close: pd.Series) -> float:
    """최근 1개월을 제외한 12개월 수익률. 거래일 기준 252일 전 → 21일 전."""
    if len(close) < 253:
        return float("nan")
    start = close.iloc[-253]
    end = close.iloc[-22]
    if start <= 0:
        return float("nan")
    return float(end / start - 1)


def compute_all(df: pd.DataFrame) -> dict:
    """일봉 DataFrame(Open/High/Low/Close/Volume)에서 필요한 지표를 한 번에 뽑습니다."""
    close = df["Close"]
    out = {
        "close": float(close.iloc[-1]),
        "prev_close": float(close.iloc[-2]) if len(close) > 1 else float("nan"),
        "ma20": _last(sma(close, 20)),
        "ma60": _last(sma(close, 60)),
        "ma200": _last(sma(close, 200)),
        "rsi14": _last(rsi(close, 14)),
        "atr14": _last(atr(df, 14)),
        "mom_12_1": momentum_12_1(close),
        "high_60": float(df["High"].tail(60).max()),
        "vol": float(df["Volume"].iloc[-1]),
        "vol_ma20": _last(sma(df["Volume"], 20)),
        "bars": int(len(df)),
        "last_date": str(df.index[-1].date()),
    }
    out["change_pct"] = (
        (out["close"] / out["prev_close"] - 1) * 100
        if out["prev_close"] == out["prev_close"] and out["prev_close"] > 0
        else float("nan")
    )
    return out


def _last(s: pd.Series) -> float:
    v = s.iloc[-1]
    return float(v) if pd.notna(v) else float("nan")
