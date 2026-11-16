"""10월 방식 지게차 규칙의 표본 외 확인 (검사용). 실행: python3 checks4.py
규칙(구역별 담당 지게차)을 10/05~21 자료만으로 정하고, 10/22~31에 적용해 실제 배정·하역 시작과 비교한다."""
import pandas as pd, numpy as np
D = "data/"; raw = pd.read_csv(D + "05_delivery_log.csv")
T = ["Reserved_Start", "Arrival_Time", "Gate_In_Time", "Unload_Start_Time", "Unload_End_Time", "Gate_Out_Time"]
for c in T: raw[c] = pd.to_datetime(raw[c], errors="coerce")
day = pd.to_datetime(raw.Date)
for n, c in zip(["rsv", "arr", "gin", "us", "ue", "go"], T): raw[n] = (raw[c] - day).dt.total_seconds() / 60
g = raw[raw.Truck_ID.str.startswith("TRK")].copy().sort_values(["Date", "gin"]).reset_index(drop=True)
g["tz"] = g.Final_Target_Zone; g["fl"] = g.Unloading_Equipment_ID; g["iw"] = g.us - g.gin
g["exg"] = (g.go - g.ue); g["exg"] = g.exg.fillna(g.exg.median()); g["unl"] = (g.ue - g.us).fillna(g.go - g.us - g.exg)
FL = ("FL-01", "FL-02", "FL-03"); TR = "2026-10-21"; tr, te = g[g.Date <= TR], g[g.Date > TR]
PREF = tr.groupby("tz").fl.agg(lambda s: s.value_counts().idxmax()).to_dict()
HAND = float(tr.iw[tr.iw <= 12].median())          # 학습 구간에서 정한 준비 시간(지게차 대기 차량 제외) 한 값
print("학습 구간(10/05~21, %d건)에서 정한 담당 지게차:" % len(tr), PREF, "| 준비 시간 중앙값:", HAND, "분")
def run(ds, hand_of):
    out = []
    for dt, x in ds.groupby("Date"):
        free = {f: 0.0 for f in FL}
        for r in x.itertuples():
            rt = r.gin + hand_of(r); idle = [k for k in FL if free[k] <= rt]
            f = PREF[r.tz] if (free[PREF[r.tz]] <= rt or not idle) else idle[0]
            us = max(rt, free[f]); free[f] = us + r.unl; out.append((r.Delivery_ID, r.fl, f, r.us, us, r.iw, us - r.gin))
    return pd.DataFrame(out, columns=["id", "fl_act", "fl_sim", "us_act", "us_sim", "iw_act", "iw_sim"])
for nm, ds in (("학습 10/05~21", tr), ("검증 10/22~31", te)):
    a = run(ds, lambda r: HAND)
    print(f"[{nm}] {len(a)}건 | 지게차 일치 {(a.fl_act == a.fl_sim).sum()}건 | 입문→하역시작: 실제 평균 {a.iw_act.mean():.2f}분 vs 모형 {a.iw_sim.mean():.2f}분"
          f" | 20분 초과: 실제 {(a.iw_act > 20).sum()}건 vs 모형 {(a.iw_sim > 20).sum()}건 | 하역 시작 오차 평균 {np.abs(a.us_act - a.us_sim).mean():.1f}분, 최대 {np.abs(a.us_act - a.us_sim).max():.0f}분")
    if nm.startswith("검증"): print("  검증 구간에서 지게차가 다른 건:", a[a.fl_act != a.fl_sim][["id", "fl_act", "fl_sim"]].values.tolist())
print("\n참고: 본 모형(bflow_sim.py)은 준비 시간을 차량별 실측값으로 넣으므로, 그쪽의 '하역 시작 80/80 일치'는 검증이 아니라 입력을 되돌려 받은 것이다.")
print("구역별 실제 배정:", pd.crosstab(g.tz, g.fl).to_dict("index"))
