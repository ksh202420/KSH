"""가상 계좌 — 신호를 그대로 따랐다면 어떻게 됐을지 계산합니다.

이 파일에서 가장 중요한 것은 수익률이 아니라 **정직함**입니다.
백테스트가 거짓말을 하는 두 가지 통로를 막아뒀습니다.

  1) 룩어헤드 — T일 종가로 만든 신호는 T+1일 시가에만 체결합니다.
     오늘 종가를 보고 오늘 종가에 샀다고 치면 현실에 없는 수익이 생깁니다.
  2) 비용 누락 — 수수료·세금·슬리피지를 전부 뺍니다.
     이걸 빼먹으면 거의 모든 전략이 좋아 보입니다.

비교 기준은 '그냥 사서 들고 있기(Buy & Hold)'입니다. 이걸 못 이기면
규칙을 늘릴 게 아니라 줄여야 합니다.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import indicators as ind
import main as M

# ── 거래 비용 가정 (보수적으로 잡습니다) ──────────────────────
FEE = 0.00015          # 매매 수수료 0.015% (양방향)
SLIPPAGE = 0.001       # 슬리피지 0.1% — 원하는 가격에 정확히 못 산다는 가정
TAX_KR_SELL = 0.0018   # 한국 매도 시 증권거래세 0.18%

WARMUP = 250           # 200일선·12개월 모멘텀이 나오려면 필요한 최소 봉 수
RISK_PCT = 0.01        # 1회 최대 손실 = 자산의 1%
MAX_WEIGHT = 0.25      # 한 종목에 자산의 25% 초과 금지
MAX_YEARS = 5
ORDER_TTL = 3          # 지정가 주문을 며칠까지 살려둘지


@dataclass
class Trade:
    code: str
    name: str
    market: str
    entry_date: str
    entry: float
    shares: float
    exit_date: str = ""
    exit: float = 0.0
    reason: str = ""
    pnl: float = 0.0
    pct: float = 0.0
    signal_date: str = ""     # 신호가 난 날 — 체결일보다 반드시 앞서야 합니다


@dataclass
class Position:
    code: str
    shares: float
    entry: float
    stop: float
    entry_date: str
    signal_date: str
    peak: float = 0.0


@dataclass
class Result:
    dates: list = field(default_factory=list)
    equity: list = field(default_factory=list)
    buyhold: list = field(default_factory=list)
    trades: list = field(default_factory=list)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """지표를 미리 전부 계산해둡니다. 하루씩 도는 루프에서는 조회만 합니다."""
    d = df.copy()
    # 소스가 섞여도 안전하도록 여기서도 시간대를 떼어냅니다.
    # (fetch.py에서 이미 맞추지만, 한 종목이라도 어긋나면 계산 전체가 멈추므로)
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    c = d["Close"]
    d["ma20"] = ind.sma(c, 20)
    d["ma200"] = ind.sma(c, 200)
    d["rsi"] = ind.rsi(c, 14)
    d["atr"] = ind.atr(d, 14)
    d["vma20"] = ind.sma(d["Volume"], 20)
    d["high60"] = d["High"].rolling(60).max()
    # 12−1 모멘텀: 252일 전 대비 21일 전 수익률
    d["mom"] = c.shift(21) / c.shift(252) - 1
    return d


def _score_row(r: pd.Series, prev_close: float) -> float:
    """main.py의 score_trend()와 같은 규칙이어야 합니다.

    규칙을 바꾸면 반드시 두 곳을 함께 바꾸십시오. 여기가 어긋나면
    가상 계좌 숫자가 화면의 판정과 다른 규칙을 검증하게 됩니다.
    """
    if pd.isna(r["ma200"]) or pd.isna(r["mom"]):
        return 0.0
    pts = M.W_MA200 if r["Close"] > r["ma200"] else -M.W_MA200
    pts += M.W_MOM if r["mom"] > 0 else -M.W_MOM
    return pts / (M.W_MA200 + M.W_MOM) * 100


def _levels(r: pd.Series):
    """진입가와 손절가. main.py의 levels()와 같은 규칙입니다.

    목표가는 두지 않습니다 — 화면과 완전히 같은 규칙으로 계산해야
    가상 계좌 숫자가 실제로 따라 할 수 있는 매매를 반영합니다.
    """
    atr = r["atr"]
    if pd.isna(atr) or atr <= 0:
        return None
    close = r["Close"]
    entry = min(close, r["ma20"]) if pd.notna(r["ma20"]) else close
    stop = entry - M.ATR_STOP * atr
    if entry - stop <= 0:
        return None
    return entry, stop


def simulate(prices: dict, meta: dict, start_cash: float = 10_000_000.0) -> Result:
    """prices: {code: OHLCV DataFrame}, meta: {code: {name, market}}"""
    prepped = {c: _prep(df) for c, df in prices.items() if len(df) > WARMUP + 5}
    if not prepped:
        return Result()

    # 모든 종목의 날짜를 합쳐 공통 달력을 만듭니다
    all_dates = sorted(set().union(*[set(d.index) for d in prepped.values()]))
    cut = all_dates[-1] - pd.DateOffset(years=MAX_YEARS)
    all_dates = [d for d in all_dates if d >= cut]

    cash = start_cash
    positions: dict[str, Position] = {}
    pending: list = []          # 다음 거래일 시가에 체결할 주문
    res = Result()

    # Buy & Hold: 첫날 전 종목에 균등 배분해 끝까지 보유
    bh_shares, bh_cash = {}, start_cash
    per = start_cash / len(prepped)
    for code, d in prepped.items():
        first = d.loc[d.index >= all_dates[0]]
        if len(first):
            px = float(first["Open"].iloc[0])
            if px > 0:
                bh_shares[code] = per / px
                bh_cash -= per

    for today in all_dates:
        # ── 1. 어제 낸 지정가 주문 처리 ─────────────────────
        #    화면이 "진입가 158에 사라"고 알려주므로, 시뮬레이션도 그 가격의
        #    지정가 주문을 넣습니다. 그 가격까지 안 내려오면 체결되지 않습니다.
        #    (시가에 아무 가격이나 사면 손익비가 화면과 달라집니다)
        still = []
        for order in pending:
            code = order["code"]
            d = prepped.get(code)
            if d is None or today not in d.index:
                still.append(order)          # 오늘 이 종목은 휴장 — 주문 유지
                continue
            if code in positions:
                continue
            order["days"] = order.get("days", 0) + 1
            op, low = float(d.loc[today, "Open"]), float(d.loc[today, "Low"])
            limit = order["entry"]
            if op <= 0 or low > limit:       # 지정가까지 안 내려옴
                if order["days"] < ORDER_TTL:
                    still.append(order)      # 며칠 더 기다립니다
                continue
            # 지정가 이하로 열렸으면 그 시가에, 아니면 지정가에 체결
            fill = min(op, limit) * (1 + SLIPPAGE)
            risk_per_share = fill - order["stop"]
            if risk_per_share <= 0:
                continue
            equity_now = cash + sum(
                p.shares * _last_close(prepped[c], today) for c, p in positions.items())
            shares = (equity_now * RISK_PCT) / risk_per_share
            shares = min(shares, (equity_now * MAX_WEIGHT) / fill, cash / (fill * (1 + FEE)))
            if shares <= 0:
                continue
            cost = shares * fill * (1 + FEE)
            if cost > cash:
                continue
            cash -= cost
            positions[code] = Position(code, shares, fill, order["stop"],
                                       str(today.date()), order["signal_date"], peak=fill)
        pending = still

        # ── 2. 보유 종목의 손절·목표 확인 (오늘 고가·저가로) ──
        for code in list(positions):
            d = prepped.get(code)
            if d is None or today not in d.index:
                continue
            r = d.loc[today]
            p = positions[code]
            p.peak = max(p.peak, float(r["High"]))
            exit_px, why = None, ""
            # 시가가 이미 손절 아래로 갭하락한 경우가 최악입니다. 그 값으로 체결합니다.
            if float(r["Open"]) <= p.stop:
                exit_px, why = float(r["Open"]), "갭하락 손절"
            elif float(r["Low"]) <= p.stop:
                exit_px, why = p.stop, "손절"
            if exit_px is not None:
                cash += _close_position(res, p, meta, exit_px, str(today.date()), why)
                del positions[code]

        # ── 3. 오늘 종가로 신호를 만들고, 주문은 내일 시가로 미룹니다 ──
        for code, d in prepped.items():
            if today not in d.index:
                continue
            loc = d.index.get_loc(today)
            if loc < WARMUP:
                continue
            r = d.iloc[loc]
            prev_close = float(d["Close"].iloc[loc - 1])
            score = _score_row(r, prev_close)

            if code in positions:
                if score <= M.SELL:                     # 추세가 꺾이면 정리
                    pending = [o for o in pending if o["code"] != code]
                    p = positions[code]
                    nxt = _next_open(d, loc)
                    if nxt:
                        ndate, nopen = nxt
                        cash += _close_position(res, p, meta, nopen * (1 - SLIPPAGE),
                                                ndate, "신호 이탈")
                        del positions[code]
                continue

            if score < M.BUY:
                continue
            lv = _levels(r)
            if not lv:
                continue
            entry, stop = lv
            if any(o["code"] == code for o in pending):
                continue
            pending.append({"code": code, "entry": entry, "stop": stop,
                            "signal_date": str(today.date()), "days": 0})

        # ── 4. 오늘 종가 기준 자산 기록 ──────────────────────
        eq = cash + sum(p.shares * _last_close(prepped[c], today) for c, p in positions.items())
        bh = bh_cash + sum(s * _last_close(prepped[c], today) for c, s in bh_shares.items())
        res.dates.append(str(today.date()))
        res.equity.append(round(eq, 2))
        res.buyhold.append(round(bh, 2))

    return res


def _next_open(d: pd.DataFrame, loc: int):
    """다음 거래일의 (날짜, 시가). 마지막 날이면 None — 미체결로 둡니다."""
    if loc + 1 >= len(d):
        return None
    return str(d.index[loc + 1].date()), float(d["Open"].iloc[loc + 1])


def _last_close(d: pd.DataFrame, today) -> float:
    """오늘까지의 마지막 종가. 휴장이면 직전 종가를 씁니다."""
    s = d["Close"].loc[:today]
    return float(s.iloc[-1]) if len(s) else 0.0


def _close_position(res: Result, p: Position, meta: dict, px: float,
                    date: str, why: str) -> float:
    m = meta.get(p.code, {})
    tax = TAX_KR_SELL if m.get("market") == "KR" else 0.0
    proceeds = p.shares * px * (1 - FEE - tax)
    cost = p.shares * p.entry
    res.trades.append(Trade(
        code=p.code, name=m.get("name", p.code), market=m.get("market", ""),
        entry_date=p.entry_date, entry=round(p.entry, 2), shares=round(p.shares, 4),
        exit_date=date, exit=round(px, 2), reason=why,
        pnl=round(proceeds - cost, 2),
        pct=round((proceeds / cost - 1) * 100, 2) if cost > 0 else 0.0,
        signal_date=p.signal_date,
    ))
    return proceeds


def stats(res: Result, start_cash: float) -> dict:
    """성적표. 총수익률만 보지 말고 MDD와 거래 횟수를 함께 보세요."""
    if not res.equity:
        return {"ok": False}
    eq = np.array(res.equity, dtype=float)
    bh = np.array(res.buyhold, dtype=float)
    years = max(len(eq) / 252, 1e-9)

    wins = [t for t in res.trades if t.pnl > 0]
    losses = [t for t in res.trades if t.pnl <= 0]
    avg_w = np.mean([t.pct for t in wins]) if wins else 0.0
    avg_l = np.mean([t.pct for t in losses]) if losses else 0.0

    return {
        "ok": True,
        "start": start_cash,
        "final": round(float(eq[-1]), 0),
        "total_pct": round(float(eq[-1] / start_cash - 1) * 100, 2),
        "cagr": round((float(eq[-1] / start_cash) ** (1 / years) - 1) * 100, 2),
        "mdd": round(float(((eq / np.maximum.accumulate(eq)) - 1).min()) * 100, 2),
        "bh_total_pct": round(float(bh[-1] / start_cash - 1) * 100, 2),
        "bh_mdd": round(float(((bh / np.maximum.accumulate(bh)) - 1).min()) * 100, 2),
        "trades": len(res.trades),
        "open_positions": 0,
        "win_rate": round(len(wins) / len(res.trades) * 100, 1) if res.trades else 0.0,
        "avg_win": round(float(avg_w), 2),
        "avg_loss": round(float(avg_l), 2),
        "payoff": round(float(avg_w / abs(avg_l)), 2) if losses and avg_l != 0 else None,
        "days": len(eq),
        "from": res.dates[0],
        "to": res.dates[-1],
        "beats_buyhold": bool(eq[-1] > bh[-1]),
    }
