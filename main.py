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
ATR_STOP = 2.0        # 손절 = 진입가 − 2.0 × ATR

# 목표가에 도달하면 파는 규칙은 2026-08-31 실험에서 제거했습니다.
# 8개 데이터셋에서 목표가 매도를 넣으면 평균 +8.99%(단순보유 이긴 횟수 1/8),
# 빼면 평균 +72.29%(5/8)로 갈렸습니다. 추세를 끝까지 타지 못하고
# 오르는 종목을 일찍 팔아버리는 것이 이 시스템의 가장 큰 손실 원인이었습니다.
# 60일 고점은 이제 '참고용 저항'으로만 보여주고, 매도는
#   ① 손절선 도달  ② 점수가 −25 이하로 떨어짐
# 두 가지로만 합니다.


# ── 점수 규칙 ─────────────────────────────────────────────────
# 2026-08-31 실험 결과 규칙을 둘로 줄였습니다.
# 10개 데이터셋에서 만점 대비 정규화해 비교한 평균 수익률:
#   전부(200일선+모멘텀+20일선+RSI+거래량)  +76.3%  거래 52회  단순보유 이김 1/10
#   RSI 제거                              +64.0%  거래 104회 0/10
#   RSI+거래량 제거                        +63.5%  거래 94회  0/10
#   200일선+모멘텀만                       +90.6%  거래 38회  2/10  ← 채택
# 규칙을 뺄수록 거래가 줄고 성적이 좋아졌습니다. 진짜 문제는 잦은 거래였습니다.
#
# 남긴 두 규칙만 논문 근거가 있습니다.
#   200일선  — Faber(2007) A Quantitative Approach to Tactical Asset Allocation
#   12−1 모멘텀 — Jegadeesh & Titman(1993), Moskowitz·Ooi·Pedersen(2012)
# RSI·20일선·거래량은 관행일 뿐 근거가 없어 뺐습니다.
# (지표 자체는 화면에 계속 보여줍니다. 점수에만 안 들어갑니다.)
W_MA200 = 30.0
W_MOM = 25.0
_FULL = W_MA200 + W_MOM     # 만점. 이 비율로 정규화해 −100~+100으로 만듭니다.


def score_trend(m: dict) -> tuple[float, list[str]]:
    """추세 점수(−100~+100)와 근거 문장.

    나올 수 있는 값은 사실상 네 가지입니다.
      +100  장기 추세도 상승, 12개월 모멘텀도 양수  → 적극매수
       +9   200일선 위인데 모멘텀은 음수            → 관망
       −9   200일선 아래인데 모멘텀은 양수          → 관망
      −100  둘 다 음수                            → 매도
    두 지표가 엇갈리면 판단하지 않는다 — 이게 이 규칙의 핵심입니다.
    """
    ma200, mom, close = m["ma200"], m["mom_12_1"], m["close"]

    # 둘 중 하나라도 못 구하면 아예 판단하지 않습니다.
    # 200일선은 이 시스템의 뼈대라서, 없으면 나머지로 때울 수 없습니다.
    if not _ok(ma200) or not _ok(mom):
        why = []
        if not _ok(ma200):
            why.append("200일선을 구할 수 없습니다 (200봉 이상 필요)")
        if not _ok(mom):
            why.append("12−1 모멘텀을 구할 수 없습니다 (253봉 이상 필요)")
        why.append("핵심 지표가 없어 판단을 보류합니다")
        return 0.0, why

    pts, why = 0.0, []
    if close > ma200:
        pts += W_MA200
        why.append("200일선 위 — 장기 추세 상승 (+30)")
    else:
        pts -= W_MA200
        why.append("200일선 아래 — 장기 추세 하락 (−30)")

    if mom > 0:
        pts += W_MOM
        why.append(f"12−1 모멘텀 +{mom*100:.0f}% — 1년 흐름 상승 (+25)")
    else:
        pts -= W_MOM
        why.append(f"12−1 모멘텀 {mom*100:.0f}% — 1년 흐름 하락 (−25)")

    score = pts / _FULL * 100
    if abs(score) < 50:
        why.append("두 지표가 서로 엇갈립니다 — 어느 쪽도 확신할 수 없어 관망")
    return score, why


def levels(m: dict) -> dict:
    """ATR 기반 진입가와 손절가. 목표가는 두지 않습니다(위 주석 참고).

    60일 고점은 '다음 저항'으로 화면에만 표시합니다 — 거기서 자동으로 팔지 않습니다.
    """
    close, ma20, atr14 = m["close"], m["ma20"], m["atr14"]
    if not _ok(atr14) or atr14 <= 0:
        return {"entry": None, "stop": None, "resistance": None, "risk_pct": None}

    entry = min(close, ma20) if _ok(ma20) else close
    stop = entry - ATR_STOP * atr14
    resistance = m["high_60"] if m["high_60"] > entry * 1.005 else None
    risk_pct = (entry - stop) / entry * 100 if entry > 0 else None
    return {"entry": entry, "stop": stop, "resistance": resistance, "risk_pct": risk_pct}


def verdict(score: float, lv: dict) -> tuple[str, str | None]:
    """점수로 판정합니다. 타점을 못 잡는 경우에만 관망으로 강등합니다."""
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

    if v in ("적극매수", "분할매수") and lv.get("entry") is None:
        return "관망", "타점 계산 불가 (ATR 없음)"
    return v, None


HIST_DAYS = 120   # 화면 그래프에 그릴 일수
START_CASH = 10_000_000.0   # 가상 계좌 초기 자금 (1천만원)
frames: dict = {}           # 시뮬레이션에 넘길 원본 일봉
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
        "hist_dates": [d.strftime("%m-%d") for d in tail.index],
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
        "resistance": _r(lv["resistance"]),
        "risk_pct": _r(lv["risk_pct"], 1),
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

    frames.clear()
    for item in watchlist:
        try:
            if selftest:
                df = _synthetic(item["code"])
            else:
                import fetch
                df = fetch.fetch(item["market"], item["code"])
                time.sleep(1.5)  # 소스를 배려한 간격. 지우지 마세요.
            frames[item["code"]] = df
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
                "stop": None, "resistance": None, "risk_pct": None,
                "demoted_reason": None, "last_date": None, "bars": 0,
                "hist": [], "hist_ma200": [], "hist_from": None, "hist_dates": [],
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


def save_paper() -> None:
    """가상 계좌 — 신호를 그대로 따랐다면 어땠을지 계산해 저장합니다."""
    import backtest

    watchlist = json.load(open("watchlist.json", encoding="utf-8"))
    meta = {i["code"]: {"name": i["name"], "market": i["market"]} for i in watchlist}
    usable = {c: d for c, d in frames.items() if c in meta}
    if len(usable) < 1:
        raise RuntimeError("시뮬레이션에 쓸 종목 데이터가 없습니다")

    res = backtest.simulate(usable, meta, START_CASH)
    st = backtest.stats(res, START_CASH)
    if not st.get("ok"):
        raise RuntimeError("시뮬레이션 결과가 비어 있습니다")

    # 자산 곡선은 매일 한 점이면 충분합니다. 거래 내역은 최근 것부터.
    trades = [t.__dict__ for t in res.trades]
    trades.sort(key=lambda t: t["exit_date"] or t["entry_date"], reverse=True)

    payload = {
        "generated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "assumptions": {
            "start_cash": START_CASH,
            "fee": backtest.FEE, "slippage": backtest.SLIPPAGE,
            "tax_kr": backtest.TAX_KR_SELL, "risk_pct": backtest.RISK_PCT,
            "max_weight": backtest.MAX_WEIGHT, "order_ttl": backtest.ORDER_TTL,
            "symbols": len(usable),
        },
        "stats": st,
        "dates": res.dates, "equity": res.equity, "buyhold": res.buyhold,
        "trades": trades[:200],
    }
    with open("docs/data/paper.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"가상 계좌 계산 완료 → 전략 {st['total_pct']:+.2f}% / "
          f"단순보유 {st['bh_total_pct']:+.2f}% / 거래 {st['trades']}회")


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
    """숫자이고 NaN이 아니면 참.

    예전에는 float만 인정해서 정수가 들어오면 '결측'으로 처리했습니다.
    지표 계산은 늘 float를 주므로 실제로는 안 걸렸지만, 값이 하나라도
    정수로 새어 들어오면 종목 전체가 조용히 관망 처리되는 버그였습니다.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return not math.isnan(float(v))


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

    if not st and os.environ.get("RUN_BACKTEST") == "1":
        try:
            save_paper()
        except Exception as e:  # noqa: BLE001
            print(f"가상 계좌 계산 실패 — 기존 결과를 그대로 둡니다: {type(e).__name__}: {e}")

    if not st and os.environ.get("RUN_DEBATE") == "1":
        try:
            import debate
            debate.run()
        except Exception as e:  # noqa: BLE001
            print(f"토론 생성 실패 — 기존 결과를 그대로 둡니다: {type(e).__name__}: {e}")

    if not st and os.environ.get("UPDATE_SYMBOLS") == "1":
        try:
            save_symbols()
        except Exception as e:  # noqa: BLE001
            print(f"종목 목록 갱신 실패 — 기존 파일을 그대로 둡니다: {type(e).__name__}: {e}")
