"""매일 한 번 실행되는 본체.

watchlist.json을 읽어 종목별로 일봉을 받고, 추세 점수와 타점을 계산해
docs/data/latest.json과 data/history/<날짜>.json 두 곳에 저장합니다.

로컬에서 테스트하려면:  python main.py
네트워크 없이 계산만 확인하려면:  python main.py --selftest
"""

import json
import math
import sys
import time
from datetime import datetime, timezone, timedelta

import pandas as pd

import indicators as ind

KST = timezone(timedelta(hours=9))

# ── 판정 임계값 ────────────────────────────────────────────────
STRONG_BUY, BUY, SELL, STRONG_SELL = 60, 25, -25, -60
MIN_RR = 1.5          # 손익비가 이보다 낮으면 매수 신호를 관망으로 강등
ATR_STOP = 2.0        # 손절 = 진입가 − 2.0 × ATR
ATR_FALLBACK_TGT = 3.0  # 위쪽에 저항이 없을 때 쓰는 목표 배수


def score_trend(m: dict) -> tuple[float, list[str]]:
    """추세 레이어 점수(−100~+100)와 사람이 읽을 근거 문장을 함께 돌려줍니다."""
    pts, why = 0.0, []
    close, ma20, ma200 = m["close"], m["ma20"], m["ma200"]

    above_200 = _ok(ma200) and close > ma200
    if _ok(ma200):
        if above_200:
            pts += 30; why.append("200일선 위 (+30)")
        else:
            pts -= 30; why.append("200일선 아래 (−30)")
    else:
        why.append("200일선 계산에 데이터 부족 (0)")

    if _ok(m["mom_12_1"]):
        if m["mom_12_1"] > 0:
            pts += 25; why.append(f"12−1 모멘텀 +{m['mom_12_1']*100:.0f}% (+25)")
        else:
            pts -= 25; why.append(f"12−1 모멘텀 {m['mom_12_1']*100:.0f}% (−25)")
    else:
        why.append("모멘텀 계산에 데이터 부족 (0)")

    if _ok(ma20):
        if close > ma20:
            pts += 15; why.append("20일선 위 (+15)")
        else:
            pts -= 15; why.append("20일선 아래 (−15)")

    r = m["rsi14"]
    if _ok(r):
        if above_200 and 30 <= r <= 45:
            pts += 20; why.append(f"상승추세 중 RSI {r:.0f} 되돌림 (+20)")
        elif r > 75:
            pts -= 15; why.append(f"RSI {r:.0f} 과열 (−15)")
        elif (not above_200) and r < 30:
            pts -= 10; why.append(f"하락추세 중 RSI {r:.0f} — 떨어지는 칼 (−10)")
        else:
            why.append(f"RSI {r:.0f} 중립 (0)")

    if _ok(m["vol_ma20"]) and m["vol_ma20"] > 0 and _ok(m["change_pct"]):
        if m["vol"] > m["vol_ma20"] * 1.5 and m["change_pct"] > 0:
            pts += 10; why.append("거래량 20일평균 1.5배 + 상승 (+10)")

    return max(-100.0, min(100.0, pts)), why


def levels(m: dict) -> dict:
    """ATR 기반 진입·손절·목표가. 목표는 직전 60일 고점(저항)을 우선 사용합니다."""
    close, ma20, atr14 = m["close"], m["ma20"], m["atr14"]
    if not _ok(atr14) or atr14 <= 0:
        return {"entry": None, "stop": None, "target": None, "rr": None,
                "target_basis": None}

    entry = min(close, ma20) if _ok(ma20) else close
    stop = entry - ATR_STOP * atr14
    resistance = m["high_60"]
    if resistance > entry * 1.005:      # 위쪽에 의미 있는 저항이 있을 때
        target, basis = resistance, "60일 고점"
    else:                                # 신고가 근처면 저항이 없으므로 ATR 배수
        target, basis = entry + ATR_FALLBACK_TGT * atr14, "ATR 3배"

    risk = entry - stop
    rr = (target - entry) / risk if risk > 0 else None
    return {"entry": entry, "stop": stop, "target": target, "rr": rr,
            "target_basis": basis}


def verdict(score: float, lv: dict) -> tuple[str, str | None]:
    """점수와 손익비로 최종 판정. 손익비가 나쁘면 매수 신호를 강등합니다."""
    if score >= STRONG_BUY:
        v = "적극매수"
    elif score >= BUY:
        v = "분할매수"
    elif score <= STRONG_SELL:
        v = "매도"
    elif score <= SELL:
        v = "비중축소"
    else:
        v = "관망"

    if v in ("적극매수", "분할매수"):
        rr = lv.get("rr")
        if rr is None:
            return "관망", "타점 계산 불가 (ATR 없음)"
        if rr < MIN_RR:
            return "관망", f"손익비 {rr:.2f} < {MIN_RR} — 신호는 있으나 자리가 나쁨"
    return v, None


HIST_DAYS = 120   # 화면 그래프에 그릴 일수


def _series(s: pd.Series, krw: bool) -> list:
    """그래프용 숫자 배열. 원화는 정수로 줄여 파일 크기를 아낍니다."""
    out = []
    for v in s:
        if pd.isna(v):
            out.append(None)
        else:
            out.append(int(round(v)) if krw else round(float(v), 2))
    return out


def analyze(item: dict, df: pd.DataFrame) -> dict:
    m = ind.compute_all(df)
    score, why = score_trend(m)
    lv = levels(m)
    v, demoted = verdict(score, lv)
    krw = item["market"] == "KR"
    tail = df["Close"].tail(HIST_DAYS)
    return {
        "hist": _series(tail, krw),
        "hist_ma200": _series(ind.sma(df["Close"], 200).tail(HIST_DAYS), krw),
        "hist_from": str(tail.index[0].date()),
        "name": item["name"],
        "code": item["code"],
        "market": item["market"],
        "currency": "USD" if item["market"] == "US" else "KRW",
        "last_date": m["last_date"],
        "bars": m["bars"],
        "close": _r(m["close"]),
        "change_pct": _r(m["change_pct"], 2),
        "score": round(score),
        "verdict": v,
        "demoted_reason": demoted,
        "reasons": why,
        "entry": _r(lv["entry"]),
        "stop": _r(lv["stop"]),
        "target": _r(lv["target"]),
        "target_basis": lv["target_basis"],
        "rr": _r(lv["rr"], 2),
        "ma20": _r(m["ma20"]),
        "ma200": _r(m["ma200"]),
        "rsi14": _r(m["rsi14"], 1),
        "atr14": _r(m["atr14"]),
        "stale": False,
        "error": None,
    }


def run(selftest: bool = False) -> dict:
    watchlist = json.load(open("watchlist.json", encoding="utf-8"))
    results, failures = [], []

    for item in watchlist:
        try:
            if selftest:
                df = _synthetic(item["code"])
            else:
                import fetch
                df = fetch.fetch(item["market"], item["code"])
                time.sleep(1.5)  # 소스를 배려한 간격. 지우지 마세요.
            row = analyze(item, df)
            results.append(row)
            print(f"  OK   {item['name']:<20} {row['score']:>4}점  {row['verdict']}")
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            failures.append({"name": item["name"], "code": item["code"], "error": msg})
            # 실패해도 성공 레코드와 같은 모양을 유지합니다.
            # 화면과 나중의 이력 분석이 필드 유무를 신경 쓰지 않아도 되도록.
            results.append({
                "name": item["name"], "code": item["code"], "market": item["market"],
                "currency": "USD" if item["market"] == "US" else "KRW",
                "verdict": "데이터 없음", "score": None, "stale": True, "error": msg,
                "reasons": [], "close": None, "change_pct": None, "entry": None,
                "stop": None, "target": None, "target_basis": None, "rr": None,
                "demoted_reason": None, "last_date": None, "bars": 0,
                "hist": [], "hist_ma200": [], "hist_from": None,
            })
            print(f"  FAIL {item['name']:<20} {msg}")

    now = datetime.now(KST)
    payload = {
        "generated_at": now.isoformat(timespec="seconds"),
        "generated_at_display": now.strftime("%Y-%m-%d %H:%M KST"),
        "total": len(watchlist),
        "ok": len(watchlist) - len(failures),
        "failed": len(failures),
        "failures": failures,
        "items": sorted(results, key=lambda x: (x["score"] is None, -(x["score"] or 0))),
    }
    return payload


def save(payload: dict) -> None:
    import os
    os.makedirs("docs/data", exist_ok=True)
    os.makedirs("data/history", exist_ok=True)
    day = payload["generated_at"][:10]
    for path in ("docs/data/latest.json", f"data/history/{day}.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n저장 완료 → docs/data/latest.json, data/history/{day}.json")


# ── 보조 ──────────────────────────────────────────────────────
def _ok(v) -> bool:
    return v is not None and isinstance(v, float) and not math.isnan(v)


def _r(v, nd: int = 2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), nd)


def _synthetic(seed_text: str) -> pd.DataFrame:
    """네트워크 없이 파이프라인을 점검하기 위한 가짜 일봉."""
    import numpy as np
    rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=400)
    n = len(idx)
    drift = rng.choice([0.0006, -0.0004])
    ret = rng.normal(drift, 0.015, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + abs(rng.normal(0, 0.008, n)))
    low = close * (1 - abs(rng.normal(0, 0.008, n)))
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1e5, 1e7, n)}, index=idx)


if __name__ == "__main__":
    st = "--selftest" in sys.argv
    print("자체 점검 모드 (가짜 데이터)" if st else "실행 시작")
    save(run(selftest=st))
