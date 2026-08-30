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
NEWS_N = 5        # 종목당 뉴스 개수


def fetch_news(item: dict, n: int = NEWS_N) -> list:
    """구글 뉴스 RSS에서 종목 관련 기사 제목을 가져옵니다.

    실패해도 예외를 밖으로 내보내지 않습니다 — 뉴스가 없는 것은 불편이지만
    분석이 멈추는 것은 사고이기 때문입니다.
    """
    import urllib.parse
    import xml.etree.ElementTree as ET

    import requests

    q = (item.get("name") or item["code"]).strip()
    url = ("https://news.google.com/rss/search?q="
           + urllib.parse.quote(q) + "&hl=ko&gl=KR&ceid=KR:ko")
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.content)

    out = []
    for el in root.iter("item"):
        title = (el.findtext("title") or "").strip()
        link = (el.findtext("link") or "").strip()
        if not title or not link:
            continue
        src = el.find("source")
        pub = (el.findtext("pubDate") or "").strip()
        out.append({
            "t": title,
            "u": link,
            "s": (src.text or "").strip() if src is not None else "",
            "d": _news_date(pub),
        })
        if len(out) >= n:
            break
    return out


def _news_date(pub: str) -> str:
    """'Fri, 29 Aug 2026 07:12:00 GMT' → '08-29'"""
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(pub).astimezone(KST).strftime("%m-%d")
    except Exception:  # noqa: BLE001
        return ""


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


def _previous_news() -> dict:
    """직전 결과에서 뉴스를 가져옵니다.

    장중 실행에서는 뉴스를 새로 받지 않고 이걸 그대로 물려줍니다.
    1시간마다 뉴스를 긁으면 구글에 부담만 주고 얻는 게 거의 없습니다.
    """
    try:
        prev = json.load(open("docs/data/latest.json", encoding="utf-8"))
        return {i["code"]: i.get("news") or [] for i in prev.get("items", [])}
    except Exception:  # noqa: BLE001
        return {}


def run(selftest: bool = False) -> dict:
    import os

    watchlist = json.load(open("watchlist.json", encoding="utf-8"))
    results, failures = [], []
    want_news = (not selftest) and os.environ.get("FETCH_NEWS") == "1"
    carried = {} if want_news else _previous_news()
    if want_news:
        print("뉴스도 함께 받습니다")

    for item in watchlist:
        try:
            if selftest:
                df = _synthetic(item["code"])
            else:
                import fetch
                df = fetch.fetch(item["market"], item["code"])
                time.sleep(1.5)  # 소스를 배려한 간격. 지우지 마세요.
            row = analyze(item, df)
            row["news"] = _news_for(item, want_news, carried)
            results.append(row)
            print(f"  OK   {item['name']:<20} {row['score']:>4}점  {row['verdict']}"
                  + (f"  뉴스 {len(row['news'])}건" if row["news"] else ""))
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
                "news": _news_for(item, want_news, carried),
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
    """결과를 저장합니다.

    latest.json은 매번 갱신하고, 날짜별 이력은 종가 확정 후에만 남깁니다.
    장중에 1시간마다 이력을 쌓으면 리포만 무겁고 얻는 게 없기 때문입니다.
    """
    import os
    import shutil

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    written = ["docs/data/latest.json"]

    # 종목 관리 화면이 현재 목록을 읽을 수 있도록 docs 안에 사본을 둡니다
    shutil.copyfile("watchlist.json", "docs/data/watchlist.json")
    written.append("docs/data/watchlist.json")

    if os.environ.get("SAVE_HISTORY", "1") == "1":
        os.makedirs("data/history", exist_ok=True)
        day = payload["generated_at"][:10]
        path = f"data/history/{day}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        written.append(path)

    print("\n저장 완료 → " + ", ".join(written))


def save_symbols() -> None:
    """종목 관리 화면의 검색에 쓸 한국 상장종목 목록을 만듭니다.

    실패해도 기존 파일을 그대로 두고 넘어갑니다 — 검색 목록이 하루 낡는 것보다
    분석이 멈추는 쪽이 훨씬 나쁘기 때문입니다.
    """
    import FinanceDataReader as fdr

    df = fdr.StockListing("KRX")
    cols = {c.lower(): c for c in df.columns}
    code_col = cols.get("code") or cols.get("symbol")
    name_col = cols.get("name")
    if not code_col or not name_col:
        raise RuntimeError(f"예상과 다른 컬럼 구성: {list(df.columns)}")

    rows = []
    for code, name in zip(df[code_col], df[name_col]):
        code = str(code).strip()
        if len(code) == 6 and code.isdigit() and isinstance(name, str) and name.strip():
            rows.append({"c": code, "n": name.strip()})

    rows.sort(key=lambda r: r["n"])
    with open("docs/data/symbols.json", "w", encoding="utf-8") as f:
        json.dump({"updated": datetime.now(KST).strftime("%Y-%m-%d"), "kr": rows},
                  f, ensure_ascii=False, separators=(",", ":"))
    print(f"한국 종목 목록 갱신 → {len(rows)}종목")


# ── 보조 ──────────────────────────────────────────────────────
def _news_for(item: dict, want: bool, carried: dict) -> list:
    """뉴스를 새로 받거나(want=True) 직전 결과에서 물려받습니다."""
    if not want:
        return carried.get(item["code"], [])
    try:
        news = fetch_news(item)
        time.sleep(0.8)          # 구글 뉴스에 대한 최소한의 예의
        return news
    except Exception as e:  # noqa: BLE001
        print(f"       ↳ {item['code']}: 뉴스 실패 ({type(e).__name__}) — 건너뜁니다")
        return carried.get(item["code"], [])


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
    import os

    st = "--selftest" in sys.argv
    print("자체 점검 모드 (가짜 데이터)" if st else "실행 시작")
    save(run(selftest=st))

    if not st and os.environ.get("UPDATE_SYMBOLS") == "1":
        try:
            save_symbols()
        except Exception as e:  # noqa: BLE001
            print(f"종목 목록 갱신 실패 — 기존 파일을 그대로 둡니다: {type(e).__name__}: {e}")
