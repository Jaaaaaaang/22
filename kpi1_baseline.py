"""KPI 1: 적치장·통로의 지적(09)·아차사고(10) 발생빈도와 조치 소요일 — 10월 기준값과 '후' 값 산출 가능성 확인.
실행: python3 kpi1_baseline.py"""
import pandas as pd, numpy as np, warnings
import assumptions as ASM
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
D = "data/"; ASOF = pd.Timestamp("2026-10-31 17:00")          # 분석 기준일(과제안내서) 작업 종료 시각
i = pd.read_csv(D + "09_inspection_action_log.csv"); n = pd.read_csv(D + "10_near_miss_reports.csv")
w = pd.read_csv(D + "02_daily_work_plan.csv"); DAYS = sorted(w.Date.unique()); ND = len(DAYS)
for c in ("Issue_DateTime", "Due_DateTime", "Closed_DateTime"): i[c] = pd.to_datetime(i[c], errors="coerce")
n["Occurrence_DateTime"] = pd.to_datetime(n["Occurrence_DateTime"]); i["Date"] = i.Issue_DateTime.dt.strftime("%Y-%m-%d"); n["Date"] = n.Occurrence_DateTime.dt.strftime("%Y-%m-%d")

# ---- 분류 기준(지적유형 코드·사건유형으로 고정. 원문은 표로 함께 출력) ----
STACK = {"STACK_HEIGHT": "B3 파렛트 적치 높이·통로 폭", "MEP_STORAGE": "D2 슬리브 임시적치가 통로를 좁힘", "DUCT_STACK": "D2 덕트 자재가 케이블 동선과 근접",
         "REBAR_ROUTE": "A2 배근 자재가 주통로 점유", "REBAR_EDGE": "A1 철근 적치가 통로 가장자리", "SIGNAL_VIEW": "C2 적치물이 신호수 시야 가림",
         "PANEL_ACCESS": "D2 분전함 앞 소형자재", "GATE_LINE": "B1 보행 유도선이 자재에 가림"}
ROUTE = {"WATER_POOL": "D3 이동통로 물고임", "RAMP_ROUTE": "D1 보행 표지 훼손", "RAMP_BARRIER": "D1 보행통로 난간", "TEMP_ROUTE": "A3 동선 분리 바리케이드",
         "PED_BARRIER": "B1 분리시설 반사표지", "QUEUE_SIGN": "B1 장외대기 안내표지"}
i["grp"] = np.where(i.Finding_Code.isin(STACK), "가. 적치물이 통로·시야 침범", np.where(i.Finding_Code.isin(ROUTE), "나. 통로 시설·상태", "다. 그 밖(범위 밖)"))
NM_STACK = ("자재·통로 간섭", "철근 적치·협착", "신호수 시야제한")
n["grp"] = np.where(n.Event_Type.isin(NM_STACK), "가. 적치물이 통로·시야 침범", "다. 그 밖(범위 밖)")
# B-Flow가 직접 바꾸는 곳: B3 적치장(D자재 → B4 이전), D자재의 D2 현장 임시적치(B4에서 필요 시점 불출)
i["reach"] = np.where(i.Finding_Code.eq("STACK_HEIGHT"), "B3 적치장", np.where(i.Finding_Code.isin(["MEP_STORAGE", "DUCT_STACK", "PANEL_ACCESS"]), "D2 임시적치", "닿지 않음"))
n["reach"] = np.where(n.Event_Type.eq("자재·통로 간섭"), "D2 임시적치", "닿지 않음")

print(f"== [0] 자료: 09번 {len(i)}건(ID 중복 {i.Inspection_ID.duplicated().sum()}), 10번 {len(n)}건(ID 중복 {n.NearMiss_ID.duplicated().sum()}), 작업일 {ND}일({DAYS[0]}~{DAYS[-1]})")
print("09 완료일시 < 발생일시:", int((i.Closed_DateTime < i.Issue_DateTime).sum()), "| 상태 Closed인데 완료일시 없음:", int(((i.Status == "Closed") & i.Closed_DateTime.isna()).sum()),
      "| 완료일시 있는데 Closed 아님:", int(((i.Status != "Closed") & i.Closed_DateTime.notna()).sum()), "| 작업일 아닌 날 발생:", int((~i.Date.isin(DAYS)).sum()), "/", int((~n.Date.isin(DAYS)).sum()))
print("10번에 조치 완료일 컬럼:", [c for c in n.columns if "Closed" in c or "Due" in c] or "없음(Action_Status만 있음)")

print("\n== [1] 발생빈도 (10월, 작업일 23일)")
t = i.groupby(["grp", "Finding_Code", "Subzone"]).size().rename("건").reset_index(); t["설명"] = t.Finding_Code.map({**STACK, **ROUTE}).fillna(""); print(t.to_string(index=False))
f = i.groupby("grp").size(); m = n.groupby("grp").size()
print("\n지적:", f.to_dict(), "| 아차사고:", m.to_dict())
print("아차사고(가) 내역:", n[n.grp.str.startswith("가")][["NearMiss_ID", "Date", "Subzone", "Event_Type", "Immediate_Cause", "Action_Status"]].values.tolist())
inn = i[~i.grp.str.startswith("다")]; nn = n[n.grp.str.startswith("가")]
print(f"적치·통로 지적(가+나) {len(inn)}건 = 작업일당 {len(inn) / ND:.2f}건, 주(6일)당 {len(inn) / ND * 6:.1f}건 | 그중 '가' {f.get('가. 적치물이 통로·시야 침범', 0)}건 = 작업일당 {f.get('가. 적치물이 통로·시야 침범', 0) / ND:.2f}건")
print(f"적치·통로 아차사고(가) {len(nn)}건 = 작업일당 {len(nn) / ND:.2f}건, 주당 {len(nn) / ND * 6:.1f}건")
print("세부구역별 지적(가+나):", inn.groupby("Subzone").size().to_dict(), "| 아차사고(가):", nn.groupby("Subzone").size().to_dict())
print("B3 적치장: 지적", int((i.Subzone == "B3").sum()), "건, 아차사고", int((n.Subzone == "B3").sum()), "건 | C3(B4 예정지): 지적", int((i.Subzone == "C3").sum()), "건", i[i.Subzone == "C3"].Finding_Code.unique().tolist(), ", 아차사고", int((n.Subzone == "C3").sum()), "건")
print("반복 지적(Recurrence_Flag=Y):", inn.groupby("grp").Recurrence_Flag.apply(lambda s: f"{(s == 'Y').sum()}/{len(s)}").to_dict())
wk = inn.assign(주=pd.to_datetime(inn.Date).dt.isocalendar().week).groupby("주").size(); wd = pd.Series(pd.to_datetime(DAYS)).dt.isocalendar().week.value_counts().sort_index()
print("주별 적치·통로 지적 건수/작업일수:", {int(k): f"{int(wk.get(k, 0))}/{int(v)}" for k, v in wd.items()})

print("\n== [2] 조치 소요일 (09번. 발생 → 조치완료)")
def lead(x, nm):
    c = x[x.Closed_DateTime.notna()]; o = x[x.Closed_DateTime.isna()]
    ld = (c.Closed_DateTime - c.Issue_DateTime).dt.total_seconds() / 86400; age = (ASOF - o.Issue_DateTime).dt.total_seconds() / 86400
    late = (c.Closed_DateTime > c.Due_DateTime).sum() + (o.Due_DateTime < ASOF).sum()
    # 미조치를 포함한 중앙값(카플란-마이어): 미조치는 '10/31 17시까지는 안 끝남'으로만 반영
    T = np.r_[ld.values, age.values]; E = np.r_[np.ones(len(ld)), np.zeros(len(age))]; S = 1.0; km = None
    for tt in np.sort(np.unique(T[E == 1])):
        S *= 1 - ((T == tt) & (E == 1)).sum() / (T >= tt).sum()
        if S <= 0.5: km = tt; break
    return {"구분": nm, "건": len(x), "조치완료": len(c), "완료율%": round(100 * len(c) / len(x)) if len(x) else None,
            "완료 건 소요일 중앙값": round(ld.median(), 2) if len(c) else None, "완료 건 평균": round(ld.mean(), 2) if len(c) else None, "완료 건 최대": round(ld.max(), 2) if len(c) else None,
            "미조치": len(o), "미조치 경과일 중앙값": round(age.median(), 1) if len(o) else None, "미조치 경과일 최대": round(age.max(), 1) if len(o) else None,
            "기한 초과(완료 지연+미조치 기한 경과)": int(late), "미조치 포함 중앙값(KM)": round(km, 2) if km is not None else f"산출 불가(절반 이상 미조치, >{T.max():.0f}일)"}
rows = [lead(i, "전체 52건"), lead(inn, "적치·통로(가+나)"), lead(i[i.grp.str.startswith("가")], " 가. 적치물 침범"), lead(i[i.grp.str.startswith("나")], " 나. 통로 시설·상태"),
        lead(i[i.Subzone == "B3"], " └ B3 적치장(STACK_HEIGHT)"), lead(i[i.reach == "D2 임시적치"], " └ D2 임시적치"), lead(i[i.grp.str.startswith("다")], "다. 그 밖")]
print(pd.DataFrame(rows).T.to_string(header=False))
print("\n적치·통로 미조치 내역(10/31 17시 기준):"); o = inn[inn.Closed_DateTime.isna()].assign(경과일=lambda x: ((ASOF - x.Issue_DateTime).dt.total_seconds() / 86400).round(1))
print(o[["Inspection_ID", "Date", "Subzone", "Finding_Code", "Severity", "Status", "경과일"]].to_string(index=False))
print("심각도별 완료율(적치·통로):", inn.groupby("Severity").apply(lambda s: f"{s.Closed_DateTime.notna().sum()}/{len(s)}").to_dict())
print("아차사고(가) 후속조치 상태:", nn.Action_Status.value_counts().to_dict(), "| 전체 25건:", n.Action_Status.value_counts().to_dict())

print("\n== [3] 시뮬레이션이 내는 값으로 이 지표를 설명할 수 있는가 (10월 일별 자료)")
dl = pd.read_csv(D + "05_delivery_log.csv"); g = dl[dl.Truck_ID.str.startswith("TRK")].copy()
for c in ("Gate_In_Time", "Gate_Out_Time", "Reserved_Start", "Arrival_Time"): g[c] = pd.to_datetime(g[c])
day = pd.DataFrame(index=DAYS)
day["D자재 반입"] = g[g.Final_Target_Zone == "D"].groupby("Date").size().reindex(DAYS, fill_value=0)
day["전체 반입"] = g.groupby("Date").size().reindex(DAYS, fill_value=0)
st = np.zeros(ND)                                   # B3 적치 물량: 시뮬레이션과 같은 가정(assumptions.STAY, 평균 1.25 작업일)의 기댓값을 10월에 적용
for k, d_ in enumerate(DAYS):
    for j in range(0, max(ASM.STAY)):
        if k + j < ND: st[k + j] += day["D자재 반입"].iloc[k] * np.mean([x > j for x in ASM.STAY])   # 체류 분포(assumptions.STAY)의 기댓값
day["B3 적치 물량(가정: 체류 분포 기댓값)"] = st
def mx(x):
    ev = sorted([(a, 1) for a in x.Gate_In_Time] + [(b, -1) for b in x.Gate_Out_Time]); c = m_ = 0
    for _, k in ev: c += k; m_ = max(m_, c)
    return m_
day["B2 최대 동시 체류"] = g.groupby("Date").apply(mx).reindex(DAYS, fill_value=0)
day["B3 지적"] = i[i.Subzone == "B3"].groupby("Date").size().reindex(DAYS, fill_value=0)
day["D2 임시적치 지적"] = i[i.reach == "D2 임시적치"].groupby("Date").size().reindex(DAYS, fill_value=0)
day["적치 침범 지적(가)"] = i[i.grp.str.startswith("가")].groupby("Date").size().reindex(DAYS, fill_value=0)
day["적치·통로 지적(가+나)"] = inn.groupby("Date").size().reindex(DAYS, fill_value=0)
day["적치·통로 아차사고"] = nn.groupby("Date").size().reindex(DAYS, fill_value=0)
print(day.T.rename(columns=lambda c: c[5:]).to_string())
X = ["D자재 반입", "전체 반입", "B3 적치 물량(가정: 체류 분포 기댓값)", "B2 최대 동시 체류"]; Y = ["B3 지적", "D2 임시적치 지적", "적치 침범 지적(가)", "적치·통로 지적(가+나)", "적치·통로 아차사고"]
print("\n일별 상관계수(스피어만, n=23):"); print(day.corr(method="spearman").loc[Y, X].round(2).to_string())
h = day["D자재 반입"] >= 2
print(f"\nD자재 반입 1대 이하인 날 {int((~h).sum())}일 vs 2대 이상인 날 {int(h.sum())}일 — 작업일당 건수:")
print(pd.DataFrame({"1대 이하": day[~h][Y].mean(), "2대 이상": day[h][Y].mean()}).round(2).to_string())
print("B3 지적 5건의 날짜:", i[i.Subzone == "B3"].Date.tolist(), "| 그날 B3 적치 물량(가정):", [int(day.loc[x, 'B3 적치 물량(가정: 체류 분포 기댓값)']) for x in i[i.Subzone == 'B3'].Date],
      "| 10/16 이후 물량 범위:", int(day.loc["2026-10-16":, "B3 적치 물량(가정: 체류 분포 기댓값)"].min()), "~", int(day.loc["2026-10-16":, "B3 적치 물량(가정: 체류 분포 기댓값)"].max()), ", 지적 0건")

print("\n== [4] B-Flow가 직접 닿는 범위(상한)")
print("지적:", i.groupby("reach").size().to_dict(), "| 아차사고:", n.groupby("reach").size().to_dict())
r = i[i.reach != "닿지 않음"]; print(f"적치·통로 지적 {len(inn)}건 중 닿는 유형 {len(r)}건({100 * len(r) / len(inn):.0f}%), 아차사고(가) {len(nn)}건 중 {int((nn.reach != '닿지 않음').sum())}건")
print("\n== [5] 참고: B구역(게이트·하역장) 아차사고와 예약 편차")
g["dev"] = (g.Reserved_Start - g.Arrival_Time).dt.total_seconds() / 60; big = g[g.dev.abs() > 60]
print("예약 대비 60분 넘게 벗어나 도착한 트럭:", big[["Delivery_ID", "Date", "dev"]].values.tolist())
print("B구역 아차사고:", n[n.Zone == "B"][["NearMiss_ID", "Date", "Subzone", "Related_Event_ID", "Event_Type"]].values.tolist())
print(f"→ 큰 편차 트럭 {len(big)}대 중 아차사고 연결 {int(big.Delivery_ID.isin(n.Related_Event_ID).sum())}대, 나머지 {len(g) - len(big)}대 중 {int(g[~g.Delivery_ID.isin(big.Delivery_ID)].Delivery_ID.isin(n.Related_Event_ID).sum())}대 (모두 10/05~06)")
day.to_csv("out/kpi1_daily_october.csv"); pd.DataFrame(rows).to_csv("out/kpi1_lead_time_october.csv", index=False)
