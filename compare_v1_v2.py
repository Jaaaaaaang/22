"""v1(이전 가정) 대 v2(모순 제거 가정) 결과 비교표. 실행: python3 compare_v1_v2.py > out/compare_v1_v2.txt
v1 결과는 out/v1/에 보관(이전 커밋의 out/*와 같은 파일)."""
import numpy as np, pandas as pd, json
pd.set_option("display.width", 250)
def pair(p):
    x = pd.read_csv(p); x = x[x.period == "90일 전체"] if "period" in x else x; return x.set_index("metric")
a, b = pair("out/v1/paired_main.csv"), pair("out/paired_main.csv")
rows = []
for m, nm in (("road", "장외대기(분/대)"), ("iw", "입문→하역시작(분/대)"), ("iw20", "입문 후 20분 초과(건)"), ("d3", "동시 하역 3대 이상 일수"), ("d4", "동시 체류 4대 이상 일수"), ("sl3", "경보 기준 초과 슬롯"),
              ("risk", "위험 점수"), ("over", "적치 면적 초과일"), ("blockAC", "강풍·적설일 A·C 반입(대)"), ("hold0", "양중 대기 합(분)")):
    rows.append(dict(지표=nm, v1_전=a.loc[m, "before"], v1_후=a.loc[m, "after"], v2_전=b.loc[m, "before"], v2_후=b.loc[m, "after"]))
T = pd.DataFrame(rows).round(2); print("== 시뮬레이션 쌍비교 (90일=76작업일, 100시나리오 평균), v1 → v2\n"); print(T.to_string(index=False))
a1, a2 = [json.load(open(f"out/{d}settings_main.json")) for d in ("v1/", "")]
print("\n== 바뀐 가정 (v1 → v2)"); 
for k, v in (("게이트 점유 GATE(분)", (3, 1)), ("적치 체류 STAY(작업일)", ("1·2·3 평균 2.00", "1·1·1·2 평균 1.25")), ("반입 지수 β", ("1.0 암묵", "0.80 추정")), ("화기 지연 적용 기간", ("76일 전체", "T018 종료 예측일 11/16까지")),
             ("C01 창 밖 예약 4건", ("학습·수요 풀에 포함", "제외(10월 재현엔 포함)")), ("KPI1 β와 KPI2 β", ("0.52±0.47 vs 1.0 (불일치)", "0.52±0.47 vs 0.80, 95% 구간 겹침 검사")), ("양중 대기 시간 지수", ("지수 그대로", "β 미적용(계획인원 비례)"))):
    print(f"  {k}: {v[0]} → {v[1]}")
