"""가정값 모순 검사. 자료를 다시 읽어 assumptions.py의 값이 자료·다른 가정과 모순되지 않는지 확인한다.
실행: python3 check_assumptions.py > out/check_assumptions.txt      (하나라도 FAIL이면 종료코드 1)
결과 표는 KPI_SPEC.md에도 들어간다(kpi_spec.py가 out/check_assumptions.csv를 읽음)."""
import sys, os, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
import assumptions as A
warnings.filterwarnings("ignore")
D = "data/"; rows = []
def chk(id_, name, ok, detail, hard=True):
    rows.append(dict(ID=id_, 검사=name, 결과="PASS" if ok else ("FAIL" if hard else "WARN"), 근거=detail))

r5 = pd.read_csv(D + "05_delivery_log.csv"); r5["day"] = pd.to_datetime(r5.Date)
for c in ("Arrival_Time", "Gate_In_Time", "Unload_Start_Time", "Unload_End_Time", "Gate_Out_Time", "Reserved_Start"): r5[c] = pd.to_datetime(r5[c], errors="coerce")
trk = r5[r5.Truck_ID.str.startswith("TRK")].copy()

# C1 게이트 점유 시간 ≤ 관측 입문 간격 최솟값(트럭+레미콘 전체)
g = r5.dropna(subset=["Gate_In_Time"]).sort_values("Gate_In_Time"); gap = (g.groupby("Date").Gate_In_Time.diff().dt.total_seconds() / 60).dropna()
chk("C1", "게이트 점유(GATE)가 관측 입문 간격보다 길지 않다", A.GATE <= gap.min(), f"GATE={A.GATE}분, 관측 {len(gap)}개 간격 최소 {gap.min():.0f}분, 3분 미만 {int((gap < 3).sum())}건(v1의 3분이면 위반)")
# C2 알림 후 지연 = 06번 고정 지연
l6 = pd.read_csv(D + "06_lifting_log.csv"); dl = (pd.to_datetime(l6.Date + " " + l6.Actual_Start.astype(str).str[-5:], errors="coerce") - pd.to_datetime(l6.Date + " " + l6.Planned_Start.astype(str).str[-5:], errors="coerce")).dt.total_seconds() / 60
dl = dl.dropna(); chk("C2", "LAG가 06번 양중 시작 지연과 같다", set(dl.round().unique()) == {float(A.LAG)}, f"LAG={A.LAG}분, 06번 지연 값 {sorted(dl.round().unique())} (n={len(dl)})")
# C3 베이·상한·지게차 수
fl = pd.read_csv(D + "17_equipment_registry.csv"); nfl = int(fl.astype(str).apply(lambda c: c.str.startswith("FL-")).any(axis=1).sum())
ev = sorted([(t, 1) for t in trk.Gate_In_Time.dropna() for _ in [0]] + [(t, -1) for t in trk.Gate_Out_Time.dropna()])
# 동시 하역(하역 시작~끝) 최대
u = sorted([(t, 1) for t in trk.Unload_Start_Time.dropna()] + [(t, -1) for t in trk.Unload_End_Time.dropna()], key=lambda x: (x[0], x[1])); n = m = 0
for _, k in u: n += k; m = max(m, n)
chk("C3", "베이 수(W0)=지게차 수=10월 동시 하역 최대", A.BAYS_W0 == len(A.FL) == nfl == m, f"BAYS_W0={A.BAYS_W0}, FL {len(A.FL)}대, 17번 지게차 {nfl}대, 05번 동시 하역 최대 {m}대")
chk("C4", "후 방식: 상한 CAP ≤ 베이 수, 베이 후 = 전 + 1", A.CAP <= A.BAYS_FULL and A.BAYS_FULL == A.BAYS_W0 + 1, f"CAP={A.CAP} ≤ 후 베이 {A.BAYS_FULL}; 유효 동시 하역 = min(베이, CAP) = {min(A.BAYS_FULL, A.CAP)} ≤ 10월 최대 {m}")
# C5 면적·체류 일관성: 10월 D자재 반입을 그대로 재생했을 때 B3 용량 초과
days = sorted(pd.read_csv(D + "02_daily_work_plan.csv").Date.unique()); dcnt = trk[trk.Final_Target_Zone == "D"].groupby("Date").size().reindex(days, fill_value=0).values
m01 = pd.read_csv(D + "01_master_schedule.csv"); hw = m01[m01.Task_Name.astype(str).str.contains(A.HOT_KEY)]; hot_end = pd.to_datetime(hw.Forecast_Finish.fillna(hw.Planned_Finish)).max()
p11 = pd.read_csv(D + "11_work_permits.csv"); hh = p11[p11.Permit_ID.astype(str).str.startswith("PTW-H")]; qhot = float((hh.Approval_Status == "조건부승인").mean()); hs = pd.to_datetime(hh.Date).min(); he = pd.to_datetime(hh.Date).max()
hotd = [(pd.Timestamp(x) >= hs) and (pd.Timestamp(x) <= he) for x in days]
cap_loads = A.B3_AREA // A.LOAD_AREA; rng = np.random.default_rng(2026); ex = []; avg = []
def replay(stay, n=3000):
    ex = []; av = []
    for _ in range(n):
        load = np.zeros(len(days) + 10)
        for k, c in enumerate(dcnt):
            for _ in range(int(c)): load[k:k + rng.choice(stay) + (1 if hotd[k] and rng.random() < qhot else 0)] += 1
        L = load[:len(days)]; ex.append(int((L > cap_loads).sum())); av.append(L.mean())
    ex = np.array(ex); return float((ex == 0).mean()), float(ex.mean()), float(np.mean(av))
pno, mex, mav = replay(A.STAY); R = A.STAY_RULE
chk("C5", "체류(STAY)·B3 면적 짝: 10월 반입 재생 시 B3 용량 초과가 거의 없다(10월 B2에 적치 초과 징후 없음)", pno >= R["p_no_overflow_min"], f"STAY={A.STAY}, B3 {cap_loads}대분: 한 번도 안 넘을 확률 {pno:.2f}(기준 ≥{R['p_no_overflow_min']}), 초과일 평균 {mex:.2f}/23일, 평균 점유 {mav:.1f}대")
v1p, v1e, _ = replay((1, 2, 3), 1500)
chk("C5v1", "참고: v1 체류 1·2·3일은 같은 검사에서 모순이었다", v1p < R["p_no_overflow_min"], f"v1: 한 번도 안 넘을 확률 {v1p:.2f}, 초과일 평균 {v1e:.2f}/23일 → v1 값을 쓰면 모순(이 항목은 'v1이 FAIL이었음'을 확인하는 항목)")
chk("C6", "체류 평균 ≥ 1일, 최대 ≤ 규칙", np.mean(A.STAY) >= R["mean_min"] and max(A.STAY) <= R["max_days"], f"평균 {np.mean(A.STAY):.2f}일, 최대 {max(A.STAY)}일(+화기 지연 최대 1일)")
chk("C7", "면적: B4 > B3, 면적이 1대 적치면적의 정수배", A.B4_AREA > A.B3_AREA and A.B3_AREA % A.LOAD_AREA == 0 and A.B4_AREA % A.LOAD_AREA == 0, f"B3 {A.B3_AREA}㎡={A.B3_AREA // A.LOAD_AREA}대분, B4 {A.B4_AREA}㎡={A.B4_AREA // A.LOAD_AREA}대분 (자료에 면적 값 없음 → 가정)")
# C8 반입 탄력성 재추정
w = pd.read_csv(D + "02_daily_work_plan.csv"); lam = [x for x in days if x >= A.LAM0]
cnt = trk[trk.Date >= A.LAM0].groupby(["Date", "Final_Target_Zone"]).size().unstack(fill_value=0).reindex(lam, fill_value=0); pw = w.groupby(["Date", "Building"]).Planned_Workers.sum().unstack()
Rr = pd.DataFrame([dict(z=z, d=dd, y=cnt.loc[dd, z], x=np.log(pw.loc[dd, b])) for z, b in A.BM.items() for dd in lam]); X = pd.get_dummies(Rr.z, dtype=float); X["lx"] = Rr.x - Rr.groupby("z").x.transform("mean")
mm = sm.GLM(Rr.y, X, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": Rr.d.factorize()[0]}); b, ci = float(mm.params["lx"]), mm.conf_int().loc["lx"].tolist()
chk("C8", "반입 탄력성 BETA_TRK가 자료 재추정 95% 구간 안, 점추정과 0.05 이내", ci[0] <= A.BETA_TRK <= ci[1] and abs(b - A.BETA_TRK) < 0.05, f"BETA_TRK={A.BETA_TRK}, 재추정 {b:.2f} (95% {ci[0]:.2f}~{ci[1]:.2f}, n={len(Rr)})")
lo, hi = A.BETA_INC - 1.96 * A.BETA_INC_SD, A.BETA_INC + 1.96 * A.BETA_INC_SD
chk("C9", "지적 탄력성(BETA_INC)과 반입 탄력성(BETA_TRK) 구간이 겹친다(서로 모순 아님)", max(lo, ci[0]) <= min(hi, ci[1]), f"지적 {lo:.2f}~{hi:.2f}, 반입 {ci[0]:.2f}~{ci[1]:.2f}")
# C10 화기
chk("C10", "화기 지연 적용 기간이 화기 허가·공종 기간과 일치", qhot == 2 / 11 and hot_end >= he, f"QHOT={qhot:.3f}(조건부승인 {int((hh.Approval_Status == '조건부승인').sum())}/{len(hh)}), 허가 {hs:%m/%d}~{he:%m/%d}, 화기 공종 {hw.Task_ID.tolist()} Forecast_Finish {hot_end:%m/%d}까지만 적용")
# C11 C01 창 밖 제외
rs = (trk.Reserved_Start - trk.day).dt.total_seconds() / 60; inw = ((rs >= 540) & (rs <= 660)) | ((rs >= 840) & (rs <= 960))
chk("C11", "C01 창 밖 예약 트럭을 학습·수요 풀에서 제외(15번 C01)", int((~inw).sum()) == 4, f"창 밖 {int((~inw).sum())}건 {trk[~inw].Delivery_ID.tolist()} 제외, 나머지 {int(inw.sum())}건 사용")
# C12 강풍일 보류
W13 = pd.read_csv(D + "13_weather_hourly.csv"); mx = W13.groupby(W13.Date.astype(str).str[:10]).Wind_mps.max(); hd = set(l6.loc[l6.Hold_Reason.astype(str).str.contains("풍속"), "Date"].astype(str).str[:10])
chk("C12", "PH=1.0: 일 최대 10 m/s 이상인 날은 모두 풍속 보류", all(x in hd for x in mx[mx >= 10].index), f"10 m/s 이상 {int((mx >= 10).sum())}일, 그중 풍속 보류 {sum(x in hd for x in mx[mx >= 10].index)}일")
# C13 휴일·작업일
c18 = pd.read_csv(D + "18_climate_reference_90d.csv"); hol = {pd.Timestamp(x) for x in A.HOLIDAYS}
bad = [wk for wk, ws, n in zip(c18.Week, pd.to_datetime(c18.Week_Start), c18.Planned_Work_Days) if len([x for x in (ws + pd.Timedelta(days=k) for k in range(6)) if x not in hol]) != n]
chk("C13", "휴일 제외 후 주당 작업일이 18번 Planned_Work_Days와 일치", not bad, f"불일치 주 {bad or '없음'} (13주)")
# C14 예보
chk("C14", "예보: 오경보 < 적중, 범위 안", A.FA < A.HIT and A.HIT_MIN <= A.HIT <= A.HIT_MAX, f"HIT={A.HIT}({A.HIT_MIN}~{A.HIT_MAX}), FA={A.FA}")
# C15 수요 지수 단위(정보)
L16 = pd.read_csv(D + "16_lookahead_90d.csv"); w45 = L16[L16.Week == sorted(L16.Week.unique())[0]].set_index("Building").Planned_Workers
oc = w[w.Date >= A.LAM0].groupby(["Date", "Building"]).Planned_Workers.sum().groupby("Building").mean()
chk("C15", "16번(행 단위 인원)과 02번(일 인원)은 단위가 달라 절대 비교하지 않고 주간 비율만 사용", True, "W45(16번)/10월 14~31일 평균(02번): " + ", ".join(f"{b} {w45[b]:.0f}/{oc[b]:.1f}={w45[b] / oc[b]:.2f}" for b in ("1동", "포디움", "2동")) + " → 같은 단위가 아님(1.0 아님). W45 = 10월 14~31 수준으로 가정(A12)", hard=False)
# C16 10월 재현(시뮬레이션 실행 결과 읽기)
if os.path.exists("out/main_n100.txt"):
    t = open("out/main_n100.txt", encoding="utf-8").read(); ok = "일치: 80/80건, 하역 시작 시각 일치(±1분): 80/80건" in t
    chk("C16", "v2 가정으로 10월 방식 재현: 지게차 배정 80/80, 하역 시작 ±1분 80/80", ok, "out/main_n100.txt의 '[② 10월 재현]' 줄")
df = pd.DataFrame(rows); os.makedirs("out", exist_ok=True); df.to_csv("out/check_assumptions.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
for x in rows: print(f"[{x['결과']}] {x['ID']} {x['검사']}\n        {x['근거']}")
print(f"\n합계: PASS {sum(x['결과'] == 'PASS' for x in rows)}, WARN {sum(x['결과'] == 'WARN' for x in rows)}, FAIL {sum(x['결과'] == 'FAIL' for x in rows)}")
sys.exit(1 if any(x["결과"] == "FAIL" for x in rows) else 0)
