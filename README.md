# 주식 타점 신호

매일 장 마감 후 자동으로 감시 종목의 일봉을 받아 추세 점수와 매수/매도 타점을 계산하고,
결과를 `docs/` 아래 정적 페이지로 보여줍니다.

- **감시 종목 수정**: `watchlist.json` 만 고치면 됩니다.
- **결과 보기**: 리포지토리 Settings → Pages 에서 켠 주소로 접속.
- **수동 실행**: Actions 탭 → daily-signals → Run workflow.
- **계산만 확인** (네트워크 없이): `python main.py --selftest`

투자 자문이 아닙니다. 계산 결과일 뿐이며 어떤 수익도 보장하지 않습니다.
