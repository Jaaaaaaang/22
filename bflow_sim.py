"""B-Flow 전·후 비교 시뮬레이션 (수정본). 원문은 bflow_sim_original.py.
실행: N_RUN=100 python3 bflow_sim.py   |  바꾼 곳은 [수정 n] 주석으로 표시.
환경변수: N_RUN, CAP(동시 체류 상한), P_COMPLY(알림 준수율), WINDOW(own|c01), QMODE(gbr|const), GEN(day|indep), TAG"""
import os, json, time
import pandas as pd, numpy as np, warnings
from ortools.sat.python import cp_model
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import PoissonRegressor, LogisticRegression
warnings.filterwarnings("ignore")
E = os.environ.get
DIR = E("BFLOW_DIR", "./data/"); N_RUN = int(E("N_RUN", 5)); SOLVE_SEC = float(E("SOLVE_SEC", 2.0))
CAP = int(E("CAP", 2)); WINDOW = E("WINDOW", "c01"); QMODE = E("QMODE", "gbr"); GEN = E("GEN", "day"); TAG = E("TAG", "main")
VAL_REP = int(E("VAL_REP", 300))

# ---------- 가정(데이터에 없는 값) ----------
P, GATE, END, LAG = float(E("P_COMPLY", 0.8)), 3, 17 * 60, 5   # 알림 준수율, 게이트 통과 분, 하역 마감, 경보 후 지연
B4_MOVE, REL, STAY = 3, 5, (1, 2, 3)         # B4 이송 분, 불출 작업 분, 적치 일수
B3_AREA, B4_AREA, LOAD_AREA = 80, 200, 20    # ㎡ (B3·B4 면적, 차량 1대 적치 면적)
ICE_MIN, HIT, FA = 120, float(E("HIT", 0.8)), 0.05   # 결빙일 양중 제한 분, 예보 적중·오경보
FL = ("FL-01", "FL-02", "FL-03"); ZC = ("A", "C"); BM = {"A": "1동", "C": "포디움", "D": "2동"}
PREF = {"A": "FL-01", "C": "FL-03", "D": "FL-02"}      # [수정 2] 10월 구역별 담당 지게차(05번 교차표의 최빈값)
HOL = {pd.Timestamp("2026-12-25"), pd.Timestamp("2027-01-01")}
COLS = ["day", "tz", "road", "moved", "iw", "gin", "go", "fl", "held"]
GC = ["day", "tz", "fl", "pref", "rsv", "rsve", "arr", "ext", "proc", "hand", "unl", "exg", "early", "dn", "u", "z", "stay"]
rd = lambda x, k=1: {a: (round(float(b), k) if isinstance(b, (int, float, np.integer, np.floating)) else b) for a, b in x.items()}

def find(df, *keys):
    for k in keys:
        for c in df.columns:
            if k.lower() in c.lower(): return c
    raise KeyError(f"{keys} 컬럼 없음. 실제 컬럼: {list(df.columns)}")

def tdt(df, c):
    s = df[c].astype(str).str.strip(); lg = s.str.len() > 8
    o = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    o[lg] = pd.to_datetime(s[lg], errors="coerce")
    o[~lg] = pd.to_datetime(df.loc[~lg, "Date"].astype(str) + " " + s[~lg], errors="coerce")
    return o

# ---------- 05번 ----------
raw = pd.read_csv(DIR + "05_delivery_log.csv")
raw["Date"] = pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d"); raw["day"] = pd.to_datetime(raw["Date"])
RS, RE = find(raw, "Reserved_Start"), find(raw, "Reserved_End")
for n, c in [("arr", "Arrival_Time"), ("gin", "Gate_In_Time"), ("us", "Unload_Start_Time"),
             ("ue", "Unload_End_Time"), ("go", "Gate_Out_Time"), ("rsv", RS), ("rsve", RE)]:
    raw[n] = (tdt(raw, c) - raw["day"]).dt.total_seconds() / 60
isrem = raw["Truck_ID"].astype(str).str.startswith("REM")
REM = {k: sorted(g.gin.dropna().tolist()) for k, g in raw[isrem].groupby("Date")}
d = raw[~isrem].copy()
d["early"] = d.rsv - d.arr; d["ext"] = d.gin - d.arr; d["iw"] = d.us - d.gin
d["exg"] = d.go - d.ue; d["exg"] = d.exg.fillna(d.exg.median())
d["unl"] = d.ue - d.us; d["unl"] = d.unl.fillna(d.go - d.us - d.exg)      # [수정 9] 출문 시각이 있으므로 역산(평균 대입 X)
d["rsve"] = d.rsve.fillna(d.rsv + 60)
d["tz"], d["fl"], d["dn"] = d["Final_Target_Zone"], d["Unloading_Equipment_ID"], 0   # [수정 2] 'eq' → 'fl' (Series.eq 메서드와 이름 충돌)
d["pref"] = d.tz.map(PREF)
n0 = len(d); d = d.dropna(subset=["arr", "rsv", "gin", "go", "us", "tz", "fl"]).sort_values(["Date", "gin"]).reset_index(drop=True)
# [수정 4] 게이트 수속 시간: 입문 − max(도착, 예약 시작). 예약 전에 입문한 차량은 입문 − 도착
d["proc"] = np.where(d.gin >= d.rsv, d.gin - np.maximum(d.arr, d.rsv), d.gin - d.arr); PROC_MED = float(d.proc.median())
# [수정 5] 차량별 입문→하역 준비 시간(hand). 자기 지게차가 끝나자마자 시작한 차량(지게차 대기)은 나머지의 중앙값
fw = []
for r in d.itertuples():
    pv = d[(d.Date == r.Date) & (d.fl == r.fl) & (d.us < r.us)]; pe = (pv.us + pv.unl).max() if len(pv) else -1
    fw.append(bool(pe > r.gin and r.us - pe <= 2))
d["fwait"] = fw; d["hand"] = np.where(d.fwait, d.loc[~d.fwait, "iw"].median(), d.iw)
ZONES = sorted(d.tz.unique()); REF = (d.ext + d.iw).mean()
print(f"[05] 일반 트럭 {n0}건 → 사용 {len(d)}건, 목적지 {ZONES}, 레미콘 일수 {len(REM)}")
print(f"[05] 도착→입문 {d.ext.mean():.1f}분 = 예약 시작까지 대기 {(d.ext - d.proc).mean():.1f}분 + 게이트 수속 {d.proc.mean():.1f}분(중앙값 {PROC_MED:.0f}분)")
print(f"[05] 지게차 대기 차량 {d.fwait.sum()}건 {d[d.fwait][['Delivery_ID', 'Date', 'fl', 'iw']].values.tolist()}, 나머지 입문→하역시작 평균 {d[~d.fwait].iw.mean():.1f}분({d[~d.fwait].iw.min():.0f}~{d[~d.fwait].iw.max():.0f})")

w = pd.read_csv(DIR + "02_daily_work_plan.csv"); w["Date"] = w["Date"].astype(str).str[:10]
OPD_OCT = sorted(w.Date.unique()); wc = find(w, "Crew_Waiting"); tcol = find(w, "Subcontractor", "Team", "협력사")
txt = w.astype(str).agg(" ".join, axis=1)
wind_m = txt.str.contains("강풍") & (w[wc] > 0); mat_m = txt.str.contains("자재반입") & (w[wc] > 0)
HOLDS = w[wind_m].groupby("Date")[wc].sum().values; HOLDS = HOLDS if len(HOLDS) else np.array([150.0])
MAT = w.loc[mat_m, wc].sum() / len(d)
sw = [len(w[(w.Date == r.Date) & (w[tcol] == r[tcol]) & (w[wc] == 0)]) > 0 for _, r in w[wind_m].iterrows()]
SW = float(np.mean(sw)) if sw else 0.0
p11 = pd.read_csv(DIR + "11_work_permits.csv"); pid = find(p11, "Permit_ID", "Permit")
acc = [c for c in p11.columns if p11[c].astype(str).str.contains("조건부승인").any()][0]
hh = p11[p11[pid].astype(str).str.startswith("PTW-H")]; QHOT = float((hh[acc].astype(str) == "조건부승인").mean())
print(f"[02] 강풍 대기 일합계 {HOLDS}, 자재반입 확인 대기 트럭당 {MAT:.2f}분, 대체 전환율 SW={SW:.2f}({int(np.sum(sw))}/{len(sw)}건), [11] 화기 조건부승인율 {QHOT:.2f}({len(hh)}건)")

# ---------- ① 분위수 회귀 ----------
def feats(df):
    ns = [int(((df.day == r.day) & ((df.rsv - r.rsv).abs() <= 15)).sum()) - 1 for r in df.itertuples()]
    return pd.DataFrame({"h": df.rsv.values / 60, "D": (df.tz == "D").astype(int).values,
                         "dow": df.day.dt.dayofweek.values, "ns": ns})
QM = {q: GBR(loss="quantile", alpha=q, n_estimators=80, max_depth=2, min_samples_leaf=3, random_state=0) for q in (.1, .5, .9)}
QC = {}
def fitQ(df):
    X = feats(df)
    for q, m in QM.items(): m.fit(X, df.early); QC[q] = float(df.early.quantile(q))
def predQ(df):
    if QMODE == "const": return np.tile([QC[.1], QC[.5], QC[.9]], (len(df), 1))
    X = feats(df); return np.sort(np.column_stack([QM[q].predict(X) for q in (.1, .5, .9)]), axis=1)
pin = lambda y, p, q: float(np.mean(np.maximum(q * (y - p), (q - 1) * (y - p))))
TR = "2026-10-21"; dtr, dte = d[d.Date <= TR], d[d.Date > TR]
fitQ(dtr); _qm = QMODE; QMODE = "gbr"; pq = predQ(dte); QMODE = _qm; ye = dte.early.values
print(f"[① 검증] 10/22~31 {len(dte)}건: 80% 구간 포함률 {np.mean((ye >= pq[:, 0]) & (ye <= pq[:, 2])):.2f}, "
      f"중앙값 MAE {np.abs(ye - pq[:, 1]).mean():.1f}분 vs 평균 대입 {np.abs(ye - dtr.early.mean()).mean():.1f}분 vs 중앙값 대입 {np.abs(ye - dtr.early.median()).mean():.1f}분")
print("[① 검증] 핀볼손실 q10/q50/q90: 분위수 회귀 " + "/".join(f"{pin(ye, pq[:, i], q):.2f}" for i, q in enumerate((.1, .5, .9)))   # [수정 10] 기준선 비교 추가
      + " vs 입력 없는 상수 분위수 " + "/".join(f"{pin(ye, QC[q], q):.2f}" for q in (.1, .5, .9)) + f"  (모형 사용: {QMODE})")
fitQ(d)

# ---------- ③ 위험 점수 ----------
cam = pd.read_csv(DIR + "04_camera_object_log.csv")
b2 = cam[cam[find(cam, "Subzone", "Camera", "Zone")].astype(str).str.contains("B2")].copy()
tcn, fcn, pcn, acn = find(b2, "Truck_Count"), find(b2, "Forklift_Count"), find(b2, "People_Count"), find(b2, "Proximity")
b2["h"] = pd.to_datetime(b2[find(b2, "Start", "Time")], errors="coerce").dt.hour; b2["dt"] = b2[find(b2, "Date")].astype(str).str[:10]
RC = [tcn, fcn, pcn]; tr_b, te_b = b2[b2.dt <= TR], b2[b2.dt > TR]
RISK = PoissonRegressor(alpha=1e-3, max_iter=1000).fit(tr_b[RC], tr_b[acn])
print(f"[③ 검증] B2 슬롯 경보 평균: 예측 {RISK.predict(te_b[RC]).mean():.2f} vs 실제 {te_b[acn].mean():.2f} (경보는 모의 사건이라 보조 점수로만 사용)")
RISK = PoissonRegressor(alpha=1e-3, max_iter=1000).fit(b2[RC], b2[acn])
PPL = b2.groupby("h")[pcn].mean().to_dict(); PPM = b2[pcn].mean()
def lam(n, h, ppl=None): return float(RISK.predict(pd.DataFrame([[n, min(n, 3), PPL.get(h, PPM) if ppl is None else ppl]], columns=RC))[0])
# [수정 6] 원본은 기준(트럭≥1 슬롯의 평균 점수)을 만족하는 k가 없어 기본값 2가 들어갔음. 해결방안 문서의 기준 슬롯(10/19 09시 B2)으로 교체
rf = b2[(b2.dt == "2026-10-19") & (b2.h == 9)].iloc[0]; THR = float(RISK.predict(pd.DataFrame([[rf[tcn], rf[fcn], rf[pcn]]], columns=RC))[0])
EMP = b2.assign(k=b2[tcn].clip(upper=3)).groupby("k")[acn].mean().to_dict()        # 트럭 수(0,1,2,3+)별 실제 경보 평균
print(f"[③] 기준 슬롯 10/19 09시: 트럭 {rf[tcn]}·지게차 {rf[fcn]}·사람 {rf[pcn]} → 점수 {THR:.2f} | 포아송 점수(10시, 평균 인원) k=1~4: {[round(lam(k, 10), 2) for k in (1, 2, 3, 4)]}"
      f" | 실제 경보 평균(트럭 0,1,2,3+): {[round(EMP[k], 2) for k in sorted(EMP)]}")
print(f"[③] 기준 슬롯 인원({rf[pcn]}명)일 때 기준 이하 최대 대수: {max(k for k in (1, 2, 3, 4) if lam(k, 9, rf[pcn]) <= THR + 1e-9)}"
      f" | 시간대 평균 인원일 때: {({h: max([k for k in (1, 2, 3, 4) if lam(k, h) <= THR + 1e-9], default=0) for h in (9, 10, 11, 14, 15)})}")
CAPS = {h: CAP for h in range(7, 17)}
print(f"[③] 적용한 동시 체류 상한: {CAP}대 (기준 슬롯의 트럭 수. 경보 자료에서 학습한 값이 아님)")

# ---------- ④ 풍속 → 양중 보류 ----------
W13 = pd.read_csv(DIR + "13_weather_hourly.csv"); mx = W13.groupby(W13[find(W13, "Date")].astype(str).str[:10])[find(W13, "Wind")].max()
L6 = pd.read_csv(DIR + "06_lifting_log.csv")
hd = set(L6.loc[L6[find(L6, "Hold_Reason")].astype(str).str.contains("풍속"), find(L6, "Date")].astype(str).str[:10])   # [수정 3] 상태 '보류' → 사유 '풍속'(10/22는 하역 미완료 보류)
yy = [int(x in hd) for x in mx.index]
PH_LOGIT = float(LogisticRegression(C=10).fit(mx.values.reshape(-1, 1), yy).predict_proba([[10.0]])[0, 1]) if len(set(yy)) > 1 else 1.0
# [수정 7] 지상 10 m/s면 지브 보정풍속이 13.5/12.5/11.5 m/s로 세 크레인 모두 중지 기준(12/12/10) 초과 → 규칙상 보류. 10월 강풍일 3일도 모두 보류
PH = float(all(10.0 * k >= lim for k, lim in ((1.35, 12.0), (1.25, 12.0), (1.15, 10.0))))
print(f"[④] 13번 {len(mx)}일, 풍속 사유 보류일 {sum(yy)}일(일최대 10 m/s 이상 {int((mx >= 10).sum())}일, 그 외 최대 {mx[mx < 10].max()} m/s). "
      f"강풍일 보류 확률: 규칙 {PH:.2f}(적용) / 로지스틱 {PH_LOGIT:.2f}(7.3~12.4 m/s 사이 자료가 없어 참고만)")

# ---------- 하역장 모형(실행) ----------
def dm(gin, go):
    ev = sorted([(t, 1) for t in gin] + [(t, -1) for t in go]); n = mxx = 0; last = None; m3 = m4 = 0.0
    for t, k in ev:
        if last is not None:
            if n >= 3: m3 += t - last
            if n >= 4: m4 += t - last
        n += k; mxx = max(mxx, n); last = t
    return mxx, m3, m4

def slots(gin, go):                                # 시간 슬롯별 최대 동시 체류(04번 B2 Truck_Count와 같은 정의, 10월 230슬롯 전부 일치)
    gin, go = np.asarray(gin, float), np.asarray(go, float); out = []
    for h in range(7, 17):
        t = np.arange(h * 60, h * 60 + 60)[:, None]; out.append(int(((gin[None, :] <= t) & (go[None, :] > t)).sum(1).max()) if len(gin) else 0)
    return out

def risk_idx(gin, go):                             # (포아송 점수 합, 실제 경보 평균표 적용 합, 기준 슬롯 초과(3대 이상) 슬롯 수)
    ns = slots(gin, go)
    X = pd.DataFrame([[n, min(n, 3), PPL.get(h, PPM)] for h, n in zip(range(7, 17), ns)], columns=RC)
    return float(RISK.predict(X).sum()), float(sum(EMP[min(n, 3)] for n in ns)), int(sum(n >= 3 for n in ns))

def execute(df, cfg, nrel, pl):
    free = {f: 0.0 for f in FL}; bay = [0.0] * cfg["bays"]
    for _ in range(nrel):
        f = min(free, key=free.get) if cfg["pool"] else "FL-02"; free[f] = max(free[f], 480.0) + REL
    T = list(df.itertuples()); a, g0 = [], []
    for i, r in enumerate(T):
        if i in pl and r.u < P:                    # [수정 4] 계획 차량도 게이트 수속 시간이 든다. 도착 안내 시각 = 계획 입문 − 수속 중앙값
            adm = pl[i] - PROC_MED; ar = adm + r.z * max((r.q90 - r.q10) / 2.563, 0.5); a.append(ar); g0.append(max(ar, adm) + r.proc)
        else: a.append(r.arr); g0.append(r.arr + r.ext)
    cap = CAP if cfg["f3"] else 99; gos = []; rows = []; tprev = -1e9
    for j in np.argsort(g0, kind="stable"):        # [수정 8] 선착순(추월 없음) + 상한 재확인
        r = T[j]; t = max(g0[j], tprev); held = 0
        while True:
            ins = sorted(x for x in gos if x > t)
            if len(ins) < cap: break
            t = ins[len(ins) - cap] + LAG; held = 1
        tprev = t; rt = t + r.hand
        if cfg["pool"]: f = min(FL, key=free.get)                                 # ② 공용 풀: 가장 먼저 비는 지게차
        else:                                                                     # [수정 2] 10월 방식: 담당 지게차 → 사용 중이면 빈 지게차 → 모두 사용 중이면 담당 대기
            idle = [k for k in FL if free[k] <= rt]; f = r.pref if (free[r.pref] <= rt or not idle) else idle[0]
        b = int(np.argmin(bay)); us = max(rt, free[f], bay[b]); ue = us + r.unl
        free[f] = ue + (B4_MOVE if cfg["f2"] and r.tz == "D" else 0); bay[b] = ue
        go = ue + r.exg; gos.append(go)
        rows.append((r.day, r.tz, t - a[j], max(0.0, a[j] - r.arr), us - t, t, go, f, held))
    return pd.DataFrame(rows, columns=COLS)

STAT = {}
def plan(df, cfg, nrel, rem):                      # ① CP-SAT 재배정 (+② 자원, ③ 상한 제약)
    n = len(df); m = cp_model.CpModel(); gv, G, U, F, D, obj = [], [], [], [], [], []
    for i, r in enumerate(df.itertuples()):
        ue, ex, hs = int(np.ceil(r.unl)), int(np.ceil(r.exg)), int(round(r.hand))
        if WINDOW == "c01":                        # 예약 블록(오전 09~11시, 오후 14~16시 시작) 안에서 이동
            b0, b1 = (540, 660) if r.rsv < 720 else (840, 960); lo0, hi0 = min(int(r.rsv), b0), max(int(r.rsve), b1)
        else: lo0, hi0 = int(r.rsv), int(max(r.rsve, r.rsv + 30))               # 원본: 자기 예약 창 [시작, 종료]. [수정 14] 기본값을 c01로 변경(자기 창 30~45분으로는 분산 불가)
        hi = min(hi0, END - ue - hs - 1); lo = min(lo0, hi)
        g = m.NewIntVar(lo, hi, f"g{i}"); u = m.NewIntVar(lo + hs, END - ue, f"u{i}"); m.Add(u >= g + hs); gv.append(g)
        G.append(m.NewFixedSizeIntervalVar(g, GATE + int(np.clip((r.q90 - r.q10) / 4, 0, 10)), f"G{i}"))
        U.append(m.NewFixedSizeIntervalVar(u, ue, f"U{i}"))
        F.append(m.NewFixedSizeIntervalVar(u, ue + (B4_MOVE if cfg["f2"] and r.tz == "D" else 0), f"F{i}"))
        e = m.NewIntVar(0, 1440, f"e{i}"); m.Add(e == u + ue + ex); sz = m.NewIntVar(0, 1440, f"z{i}")   # [수정 1] 구간 길이는 변수 1개여야 함
        D.append(m.NewIntervalVar(g, sz, e, f"D{i}"))
        dv = m.NewIntVar(0, 1440, f"d{i}"); tg = int(np.clip(r.rsv, lo, hi)); m.Add(dv >= g - tg); m.Add(dv >= tg - g)
        obj.append(dv + 5 * (u - g - hs))
    m.AddNoOverlap(G + [m.NewFixedSizeIntervalVar(int(t), GATE, "") for t in rem])
    m.AddCumulative(U, [1] * n, cfg["bays"])
    rl = [m.NewFixedSizeIntervalVar(480 + (k // 3 if cfg["pool"] else k) * REL, REL, "") for k in range(nrel)]   # [수정 13] 원본은 전부 08:00에 겹쳐 4건 이상이면 항상 INFEASIBLE. 실행 모형과 같게 순차 배치
    m.AddCumulative(F + rl, [1] * (n + nrel), 3)
    if cfg["f3"]: m.AddCumulative(D, [1] * n, CAP)
    m.Minimize(sum(obj)); s = cp_model.CpSolver(); s.parameters.max_deterministic_time = SOLVE_SEC; s.parameters.max_time_in_seconds = 30
    s.parameters.num_workers = 1; s.parameters.random_seed = 0                    # 재현성: 단일 스레드·고정 시드·결정적 시간 제한(기계 부하와 무관)
    st = s.Solve(m); k = s.StatusName(st); c = STAT.setdefault(cfg["name"], {}); c[k] = c.get(k, 0) + 1
    return {i: s.Value(gv[i]) for i in range(n)} if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else {}

def day_run(df, cfg, nrel, rem):
    if len(df) == 0: return pd.DataFrame(columns=COLS), 0
    df = df.reset_index(drop=True); pl = {}
    if cfg["f1"]:
        q = predQ(df); df["q10"], df["q50"], df["q90"] = q[:, 0], q[:, 1], q[:, 2]; pl = plan(df, cfg, nrel, rem)
    return execute(df, cfg, nrel, pl), int(cfg["f1"] and not pl)

BASE = dict(name="W0 10월 방식", f1=False, f2=False, f3=False, f4=False, pool=False, bays=3, hand=0)   # 하역 동시 최대 3대(=지게차 수)가 10월 관측 상한
d["u"], d["z"], d["stay"] = 1.0, 0.0, 1

# ---------- 10월 재현과 검증 ----------
def summ(R, nd):                                   # R: gin·go·iw·day가 있는 표, nd: 일수
    pm = [dm(x.gin, x.go) for _, x in R.groupby("day")]
    return dict(트럭_일=len(R) / nd, iw=R.iw.mean(), iw20_100대=100 * (R.iw > 20).mean(), d3_비율=sum(p[0] >= 3 for p in pm) / nd,
                d4_비율=sum(p[0] >= 4 for p in pm) / nd, m3_일=sum(p[1] for p in pm) / nd, m4_일=sum(p[2] for p in pm) / nd)
rp = pd.concat([execute(g.reset_index(drop=True), BASE, 0, {}) for _, g in d.groupby("Date")], ignore_index=True)
print(f"\n[② 10월 재현] 10월 방식 지게차 규칙이 실제 배정과 일치: {(rp.fl.values == d.fl.values).sum()}/{len(d)}건, 하역 시작 시각 일치(±1분): {(np.abs(rp.gin + rp.iw - d.us) <= 1).sum()}/{len(d)}건")
print("  실제", rd(summ(d, d.Date.nunique()), 2)); print("  모형", rd(summ(rp, d.Date.nunique()), 2))

# ---------- 11월 이후 시나리오 ----------
L16 = pd.read_csv(DIR + "16_lookahead_90d.csv")
PW = L16.pivot_table(index=find(L16, "Week"), columns=find(L16, "Building"), values=find(L16, "Planned_Workers", "Worker", "인원"), aggfunc="sum")
C18 = pd.read_csv(DIR + "18_climate_reference_90d.csv").set_index("Week"); C18["ws"] = pd.to_datetime(C18["Week_Start"]); WEEKS = list(C18.index)
W45 = sorted(PW.index)[0]; print("16번에 없는 주:", [x for x in WEEKS if x not in PW.index])
def IDX(wk, z):
    b = BM.get(z)
    return 1.0 if b not in PW.columns else PW.loc[wk if wk in PW.index else PW.index[-1], b] / PW.loc[W45, b]
OPD = {wk: [x for x in (C18.loc[wk, "ws"] + pd.Timedelta(days=i) for i in range(6)) if x not in HOL] for wk in WEEKS}
LAM0 = "2026-10-14"; LAMD = [x for x in OPD_OCT if x >= LAM0]
cnt = d[d.Date >= LAM0].groupby(["Date", "tz"]).size().unstack(fill_value=0).reindex(index=LAMD, columns=ZONES, fill_value=0)
PAR = {z: (cnt[z].mean(), max(cnt[z].var(ddof=1), 1e-9)) for z in ZONES}; TM = {z: d[d.tz == z].reset_index(drop=True) for z in ZONES}
DAYS = {x: d[d.Date == x].reset_index(drop=True) for x in LAMD}
QPOUR = len(REM) / len(OPD_OCT)
print(f"수요 기준(10/14~31, {len(LAMD)}일): 구역별 일평균 {rd({z: PAR[z][0] for z in ZONES}, 2)}, 분산 {rd({z: PAR[z][1] for z in ZONES}, 2)}, 일 합계 평균 {cnt.sum(1).mean():.2f}·분산 {cnt.sum(1).var(ddof=1):.2f}")

def draw(m_, v_, rng):
    if m_ > 0 and v_ > m_ * 1.05: r = m_ * m_ / (v_ - m_); return rng.negative_binomial(r, r / (r + m_))
    return rng.poisson(m_)
def gen_day(day, idxf, rng, mode=None):
    mode = mode or GEN; parts = []; src = None
    if mode == "day":                              # [수정 11] 10월 하루를 통째로 뽑고, 계획인원 지수만큼 구역별로 트럭을 더하거나 뺀다
        src = LAMD[rng.integers(len(LAMD))]; base = DAYS[src]
        for z in ZONES:
            bz = base[base.tz == z]; x = len(bz) * idxf(z); n = int(x) + int(rng.random() < x - int(x))
            if n <= len(bz): parts.append(bz.iloc[np.sort(rng.permutation(len(bz))[:n])])
            else: parts.append(pd.concat([bz, TM[z].iloc[rng.integers(0, len(TM[z]), n - len(bz))]]))
    else:                                          # 원본: 구역별 대수를 독립 추출하고 트럭도 독립 추출
        for z in ZONES:
            k = idxf(z); n = draw(PAR[z][0] * k, PAR[z][1] * k, rng); parts.append(TM[z].iloc[rng.integers(0, len(TM[z]), n)])
    o = pd.concat(parts, ignore_index=True) if parts else d.iloc[:0].copy()
    o = o.assign(day=day, dn=0, u=rng.random(len(o)), z=rng.standard_normal(len(o)), stay=rng.choice(STAY, len(o)))   # [수정 12] 난수를 미리 뽑아 모든 방식이 공유
    return o[GC].reset_index(drop=True), src

act = summ(d[d.Date >= LAM0], len(LAMD)); print(f"\n[수요 생성 검증] 계획인원 지수 1.0으로 만든 가상 {len(LAMD)}일 × {VAL_REP}회에 10월 방식을 적용 → 실제 10/14~31과 비교")
print("  실제 10/14~31        ", rd(act, 2))
for md, nm in (("indep", "원본(독립 추출)   "), ("day", "수정(하루 통째 추출)")):
    rng = np.random.default_rng(777); acc_ = []
    for _ in range(VAL_REP):
        R = pd.concat([execute(gen_day(pd.Timestamp("2026-10-01") + pd.Timedelta(days=i), lambda z: 1.0, rng, md)[0], BASE, 0, {}) for i in range(len(LAMD))], ignore_index=True)
        acc_.append(summ(R, len(LAMD)))
    A = pd.DataFrame(acc_); print(f"  {nm}", {c: f"{A[c].mean():.2f} [{A[c].quantile(.05):.2f}~{A[c].quantile(.95):.2f}]" for c in A})

def make_scen(seed):
    rng = np.random.default_rng(10000 + seed); S = []
    for wk in WEEKS:
        row = C18.loc[wk]; days = OPD[wk]; n = len(days)
        def kk(x):
            x = x * n / 7; return min(n, int(x) + int(rng.random() < x - int(x)))
        wd, sn, ic = [set(rng.choice(n, kk(row[c]), replace=False)) for c in ("Expected_Strong_Wind_Days", "Expected_Snow_Days", "Freezing_Days")]
        sc = (IDX(wk, "A") + IDX(wk, "C")) / 2
        for i, day in enumerate(days):
            wind = (i in wd) and rng.random() < PH; blocked = wind or (i in sn)
            hold = rng.choice(HOLDS) * sc if blocked else (ICE_MIN * sc if i in ic else 0.0)
            hot = rng.random() < QHOT; uf = rng.random(); pour = rng.random() < QPOUR; rk = rng.choice(list(REM))
            trucks, src = gen_day(day, lambda z, wk=wk: IDX(wk, z), rng)
            rem = list(REM.get(src, [])) if GEN == "day" else (list(REM[rk]) if pour else [])   # 하루 통째 추출이면 그날의 레미콘도 함께
            S.append(dict(day=day, week=wk, blocked=blocked, wind=wind, snow=i in sn, ice=i in ic, hold=hold, hot=hot, uf=uf,
                          cold=int(row["Avg_Temp_C"] <= 4), rem=rem, trucks=trucks))
    return S

def run_policy(S, cfg):
    carry = None; pend = []; recs = []
    for i, s in enumerate(S):
        tr = s["trucks"]
        if carry is not None and len(carry): tr = pd.concat([tr, carry], ignore_index=True)
        carry = None; flag = cfg["f4"] and s["uf"] < (HIT if s["blocked"] else FA); defer = 0
        if flag: pend = [(i if due == i + 1 else due, k) for due, k in pend]          # 불출 앞당김
        nrel = sum(k for due, k in pend if due == i)
        if flag and i + 1 < len(S) and len(tr):                                       # A·C 자재 이연
            mk = tr.tz.isin(ZC) & (tr.dn < 2)
            if mk.any():
                mv = tr[mk].copy(); mv["day"] = S[i + 1]["day"]; mv["dn"] += 1; defer = len(mv); tr = tr[~mk]; carry = mv
        out, fail = day_run(tr, cfg, nrel, s["rem"])
        rec = dict(week=s["week"], n=len(out), road=0., moved=0., iw=0., iw20=0, mx=0, m3=0., m4=0., risk=0., risk_emp=0., sl3=0, held=0,
                   fail=fail, planned=int(cfg["f1"] and len(out) > 0), pour=int(len(s["rem"]) > 0), blk=int(s["blocked"]), ice=int(s["ice"] and not s["blocked"]))
        if len(out):
            mxx, m3, m4 = dm(out.gin, out.go); rk, re_, s3 = risk_idx(out.gin, out.go)
            rec.update(road=out.road.sum(), moved=out.moved.sum(), iw=out.iw.sum(), iw20=int((out.iw > 20).sum()), mx=mxx, m3=m3, m4=m4,
                       risk=rk, risk_emp=re_, sl3=s3, held=int(out.held.sum()))
        nd_ = tr[tr.tz == "D"] if len(tr) else tr
        pend = [(due, k) for due, k in pend if due > i] + [(i + int(st) + int(s["hot"]), 1) for st in (nd_.stay.values if len(nd_) else [])]
        loads = sum(k for _, k in pend); ca = B4_AREA if cfg["f2"] else B3_AREA
        rec.update(loads=loads, over=int(loads * LOAD_AREA > ca), occ=loads * LOAD_AREA / ca, cold=s["cold"],
                   blockAC=int(tr.tz.isin(ZC).sum()) if s["blocked"] and len(tr) else 0, fa_defer=0 if s["blocked"] else defer,
                   hold0=s["hold"], hold1=s["hold"] * (1 - SW) if (flag and s["blocked"]) else s["hold"])
        recs.append(rec)
    return pd.DataFrame(recs)

def agg(R):
    n = max(R.n.sum(), 1); road, iw = R.road.sum() / n, R.iw.sum() / n
    return dict(days=len(R), trucks=n, road=road, moved=R.moved.sum() / n, iw=iw, iw20=R.iw20.sum(), d3=(R.mx >= 3).sum(), d4=(R.mx >= 4).sum(),
                m3=R.m3.sum(), m4=R.m4.sum(), sl3=R.sl3.sum(), risk=R.risk.sum(), risk_emp=R.risk_emp.sum(), held=R.held.sum(),
                plan_fail=R.fail.sum(), loads_max=R.loads.max(), over=R.over.sum(), occ=R.occ.max(), blk_days=R.blk.sum(), blockAC=R.blockAC.sum(),
                fa_defer=R.fa_defer.sum(), hold0=R.hold0.sum(), hold1=R.hold1.sum(), mat=n * MAT * (road + iw) / REF, cold=R.cold.sum(), pour=R.pour.sum())

FULL = {**BASE, "name": "B-Flow 전체", "f1": True, "f2": True, "f3": True, "f4": True, "pool": True, "bays": BASE["bays"] + 1}
CFGS = {"W0 10월 방식": BASE, "B-Flow 전체": FULL, "전체-①": {**FULL, "name": "전체-①", "f1": False},
        "전체-②": {**FULL, "name": "전체-②", "f2": False, "pool": False, "bays": BASE["bays"]},
        "전체-③": {**FULL, "name": "전체-③", "f3": False}, "전체-④": {**FULL, "name": "전체-④", "f4": False}}
WIN = set(C18.index[C18["ws"] >= pd.Timestamp("2026-12-07")])
PER = {"90일 전체": lambda R: R, "동절기 전(~W49)": lambda R: R[~R.week.isin(WIN)], "동절기(W50~)": lambda R: R[R.week.isin(WIN)]}
res = {p: {k: [] for k in CFGS} for p in PER}; t0 = time.time()
for sd in range(N_RUN):
    S = make_scen(sd)
    for nm, cfg in CFGS.items():
        R = run_policy(S, cfg)
        for p, f in PER.items(): res[p][nm].append(agg(f(R)))
print(f"\n시나리오 {N_RUN}개 × 방식 {len(CFGS)}개 완료 ({time.time() - t0:.0f}초). 설정: 상한 {CAP}대, 준수율 {P}, 예약 이동 범위 {WINDOW}, 분위수 {QMODE}, 수요 생성 {GEN}")
print("CP-SAT 풀이 상태(일 단위):", STAT)
MET = ["days", "trucks", "road", "moved", "iw", "iw20", "d3", "d4", "m3", "m4", "sl3", "risk", "risk_emp", "held", "plan_fail", "loads_max", "over", "occ",
       "blk_days", "blockAC", "fa_defer", "hold0", "hold1", "mat", "cold", "pour"]
ci = lambda x: f"{np.mean(x):.1f} [{np.percentile(x, 5):.1f}~{np.percentile(x, 95):.1f}]"
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 60)
for p in PER:
    print(f"\n[{p}] 시나리오 평균 [시나리오 간 5~95% 범위]")
    print(pd.DataFrame({k: {m: ci([a[m] for a in v]) for m in MET} for k, v in res[p].items()}).to_string())
PAIRS = [("road", "road"), ("iw", "iw"), ("iw20", "iw20"), ("d3", "d3"), ("d4", "d4"), ("m3", "m3"), ("m4", "m4"), ("sl3", "sl3"), ("risk", "risk"),
         ("risk_emp", "risk_emp"), ("over", "over"), ("blockAC", "blockAC"), ("hold0", "hold1"), ("mat", "mat")]
rows = []
for p in PER:
    print(f"\n[쌍비교 {p}] W0 → B-Flow 전체 (같은 시나리오끼리 비교. 감소율 = 1 − 후 평균/전 평균, 차이 범위는 시나리오별 (전−후)의 5~95%)")
    for m, m1 in PAIRS:
        a0 = np.array([x[m] for x in res[p]["W0 10월 방식"]], float); a1 = np.array([x[m1] for x in res[p]["B-Flow 전체"]], float); df_ = a0 - a1
        pct = f"{(1 - a1.mean() / a0.mean()) * 100:.0f}%" if a0.mean() > 0 else "미산출"
        print(f"  {m:9s} 전 {a0.mean():8.2f} → 후 {a1.mean():8.2f} | 감소 {df_.mean():8.2f} [{np.percentile(df_, 5):.2f}~{np.percentile(df_, 95):.2f}] | 감소율 {pct} | 후가 더 나쁜 시나리오 {int((df_ < 0).sum())}/{len(df_)}")
        rows.append(dict(period=p, metric=m, before=a0.mean(), after=a1.mean(), diff=df_.mean(), diff_p5=np.percentile(df_, 5), diff_p95=np.percentile(df_, 95), worse=int((df_ < 0).sum()), n=len(df_)))
os.makedirs("out", exist_ok=True)
pd.DataFrame(rows).to_csv(f"out/paired_{TAG}.csv", index=False)
pd.DataFrame([dict(period=p, config=k, run=i, **{a: float(b) for a, b in v.items()}) for p in res for k in res[p] for i, v in enumerate(res[p][k])]).to_csv(f"out/runs_{TAG}.csv", index=False)
json.dump(dict(N_RUN=N_RUN, CAP=CAP, P=P, WINDOW=WINDOW, QMODE=QMODE, GEN=GEN, HIT=HIT, FA=FA, PH=PH, PH_LOGIT=PH_LOGIT, SW=SW, QHOT=QHOT, MAT=MAT, PROC_MED=PROC_MED,
               THR=THR, STAT=STAT, assumptions=dict(GATE=GATE, LAG=LAG, B4_MOVE=B4_MOVE, REL=REL, STAY=STAY, B3_AREA=B3_AREA, B4_AREA=B4_AREA, LOAD_AREA=LOAD_AREA, ICE_MIN=ICE_MIN)),
          open(f"out/settings_{TAG}.json", "w"), ensure_ascii=False, indent=1)
