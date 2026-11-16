"""사용자가 전달한 B-Flow 시뮬레이션 코드 원문 (DIR 경로와 N_RUN/SOLVE_SEC 환경변수만 변경).
검사용 기준본이며, 수정본은 bflow_sim.py 입니다."""
import os
import pandas as pd, numpy as np, warnings
from ortools.sat.python import cp_model
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import PoissonRegressor, LogisticRegression
warnings.filterwarnings("ignore")
DIR = os.environ.get("BFLOW_DIR", "./data/"); N_RUN = int(os.environ.get("N_RUN", 5)); SOLVE_SEC = 1.0      # 먼저 5회, 이상 없으면 30~200

# ---------- 가정(데이터에 없는 값) ----------
P, GATE, END, LAG = 0.8, 3, 17 * 60, 5       # 알림 준수율, 게이트 통과 분, 하역 마감, 경보 후 지연
B4_MOVE, REL, STAY = 3, 5, (1, 2, 3)         # B4 이송 분, 불출 작업 분, 적치 일수
B3_AREA, B4_AREA, LOAD_AREA = 80, 200, 20    # ㎡ (B3·B4 면적, 차량 1대 적치 면적)
ICE_MIN, HIT, FA = 120, 0.8, 0.05            # 결빙일 양중 제한 분, 예보 적중·오경보
FL = ("FL-01", "FL-02", "FL-03"); ZC = ("A", "C"); BM = {"A": "1동", "C": "포디움", "D": "2동"}
HOL = {pd.Timestamp("2026-12-25"), pd.Timestamp("2027-01-01")}
COLS = ["day", "tz", "road", "moved", "iw", "gin", "go"]
GC = ["day", "tz", "eq", "rsv", "rsve", "arr", "ext", "unl", "exg", "early", "dn"]

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
REM = {k: g.gin.dropna().tolist() for k, g in raw[isrem].groupby("Date")}
d = raw[~isrem].copy()
d["early"] = d.rsv - d.arr; d["ext"] = d.gin - d.arr; d["iw"] = d.us - d.gin
d["unl"] = d.ue - d.us; d["unl"] = d.unl.fillna(d.unl.mean())
d["exg"] = d.go - d.ue; d["exg"] = d.exg.fillna(d.exg.median())
d["rsve"] = d.rsve.fillna(d.rsv + 60)
d["tz"], d["eq"], d["dn"] = d["Final_Target_Zone"], d["Unloading_Equipment_ID"], 0
n0 = len(d); d = d.dropna(subset=["arr", "rsv", "gin", "go", "us", "tz", "eq"]).reset_index(drop=True)
ZONES = sorted(d.tz.unique()); REF = (d.ext + d.iw).mean()
print(f"[05] 일반 트럭 {n0}건 → 사용 {len(d)}건, 목적지 {ZONES}, 레미콘 일수 {len(REM)}")

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
print(f"[02] 강풍 대기 일합계 {HOLDS}, 자재반입 확인 대기 트럭당 {MAT:.2f}분, 대체 전환율 SW={SW:.2f}, [11] 화기 조건부승인율 {QHOT:.2f}")

# ---------- ① 분위수 회귀 ----------
def feats(df):
    ns = [int(((df.day == r.day) & ((df.rsv - r.rsv).abs() <= 15)).sum()) - 1 for r in df.itertuples()]
    return pd.DataFrame({"h": df.rsv.values / 60, "D": (df.tz == "D").astype(int).values,
                         "dow": df.day.dt.dayofweek.values, "ns": ns})
QM = {q: GBR(loss="quantile", alpha=q, n_estimators=80, max_depth=2, min_samples_leaf=3, random_state=0) for q in (.1, .5, .9)}
def fitQ(df):
    X = feats(df)
    for m in QM.values(): m.fit(X, df.early)
def predQ(df):
    X = feats(df); return np.sort(np.column_stack([QM[q].predict(X) for q in (.1, .5, .9)]), axis=1)
TR = "2026-10-21"; dtr, dte = d[d.Date <= TR], d[d.Date > TR]
fitQ(dtr); pq = predQ(dte)
print(f"[① 검증] 10/22~31 {len(dte)}건: 80% 구간 포함률 {np.mean((dte.early.values >= pq[:, 0]) & (dte.early.values <= pq[:, 2])):.2f}, "
      f"중앙값 MAE {np.abs(dte.early.values - pq[:, 1]).mean():.1f}분 vs 평균 대입 {np.abs(dte.early.values - dtr.early.mean()).mean():.1f}분")
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
THR = RISK.predict(b2[RC])[b2[tcn].values >= 1].mean()
def lam(n, h): return float(RISK.predict(pd.DataFrame([[n, min(n, 3), PPL.get(h, PPM)]], columns=RC))[0])
CAPS = {h: next((k for k in (4, 3, 2) if lam(k, h) <= THR), 2) for h in range(7, 17)}
print("[③] 시간대별 동시 체류 상한:", CAPS)

# ---------- ④ 풍속 → 양중 보류 ----------
W13 = pd.read_csv(DIR + "13_weather_hourly.csv"); mx = W13.groupby(W13[find(W13, "Date")].astype(str).str[:10])[find(W13, "Wind")].max()
L6 = pd.read_csv(DIR + "06_lifting_log.csv"); hd = set(L6.loc[L6[find(L6, "Status")].astype(str).str.contains("보류"), find(L6, "Date")].astype(str).str[:10])
yy = [int(x in hd) for x in mx.index]
PH = float(LogisticRegression(C=10).fit(mx.values.reshape(-1, 1), yy).predict_proba([[10.0]])[0, 1]) if len(set(yy)) > 1 else 1.0
print(f"[④] 13번 {len(mx)}일, 보류일 {sum(yy)}일, 풍속 10 m/s일 때 보류 확률 {PH:.2f}")

# ---------- 하역장 모형(실행) ----------
def dm(gin, go):
    ev = sorted([(t, 1) for t in gin] + [(t, -1) for t in go]); n = mxx = 0; last = None; m3 = m4 = 0.0
    for t, k in ev:
        if last is not None:
            if n >= 3: m3 += t - last
            if n >= 4: m4 += t - last
        n += k; mxx = max(mxx, n); last = t
    return mxx, m3, m4

def risk_idx(gin, go):
    X = []
    for h in range(7, 17):
        n = max(sum(1 for a, b in zip(gin, go) if a <= t < b) for t in range(h * 60, h * 60 + 60, 10))
        X.append([n, min(n, 3), PPL.get(h, PPM)])
    return float(RISK.predict(pd.DataFrame(X, columns=RC)).sum())

def execute(df, cfg, nrel, pl, rng):
    free = {f: 0.0 for f in FL}; bay = [0.0] * cfg["bays"]; inside = []
    for _ in range(nrel):
        f = min(free, key=free.get) if cfg["pool"] else "FL-02"; free[f] = max(free[f], 480.0) + REL
    a, g0 = [], []
    for i, r in enumerate(df.itertuples()):
        if i in pl and rng.random() < P:
            ar = pl[i] + rng.normal(0, max((r.q90 - r.q10) / 2.563, 0.5)); a.append(ar); g0.append(max(ar, pl[i]))
        else: a.append(r.arr); g0.append(r.arr + r.ext)
    rows = []
    for j in np.argsort(g0):
        r = df.iloc[j]; t = g0[j]; inside = [x for x in inside if x > t]
        while len(inside) >= (CAPS.get(int(t // 60), 4) if cfg["f3"] else 99):
            t = min(inside) + LAG; inside = [x for x in inside if x > t]
        ks = list(FL) if cfg["pool"] else [r.eq if r.eq in free else "FL-01"]
        f = min(ks, key=free.get); b = int(np.argmin(bay))
        us = max(t + cfg["hand"], free[f], bay[b]); ue = us + r.unl
        free[f] = ue + (B4_MOVE if cfg["f2"] and r.tz == "D" else 0); bay[b] = ue
        go = ue + r.exg; inside.append(go)
        rows.append((r.day, r.tz, t - a[j], max(0.0, a[j] - r.arr), us - t, t, go))
    return pd.DataFrame(rows, columns=COLS)

def plan(df, cfg, nrel, rem):                      # ① CP-SAT 재배정 (+② 자원, ③ 상한 제약)
    n = len(df); hs = int(cfg["hand"]); m = cp_model.CpModel()
    gv, G, U, F, D, obj = [], [], [], [], [], []
    for i, r in enumerate(df.itertuples()):
        ue, ex = int(np.ceil(r.unl)), int(np.ceil(r.exg))
        hi = min(int(max(r.rsve, r.rsv + 30)), END - ue - hs - 1); lo = min(int(r.rsv), hi)
        g = m.NewIntVar(lo, hi, f"g{i}"); u = m.NewIntVar(lo + hs, END - ue, f"u{i}"); m.Add(u >= g + hs); gv.append(g)
        G.append(m.NewFixedSizeIntervalVar(g, GATE + int(np.clip((r.q90 - r.q10) / 4, 0, 10)), f"G{i}"))
        U.append(m.NewFixedSizeIntervalVar(u, ue, f"U{i}"))
        F.append(m.NewFixedSizeIntervalVar(u, ue + (B4_MOVE if cfg["f2"] and r.tz == "D" else 0), f"F{i}"))
        D.append(m.NewIntervalVar(g, u + ue + ex - g, u + ue + ex, f"D{i}"))
        obj.append((g - lo) + 5 * (u - g - hs))
    m.AddNoOverlap(G + [m.NewFixedSizeIntervalVar(int(t), GATE, "") for t in rem])
    m.AddCumulative(U, [1] * n, cfg["bays"])
    rl = [m.NewFixedSizeIntervalVar(480, REL, "") for _ in range(nrel)]
    m.AddCumulative(F + rl, [1] * (n + nrel), 3)
    if cfg["f3"]:
        cm = max(CAPS.values()); hl = [h for h in CAPS if CAPS[h] < cm]
        m.AddCumulative(D + [m.NewFixedSizeIntervalVar(h * 60, 60, "") for h in hl], [1] * n + [cm - CAPS[h] for h in hl], cm)
    m.Minimize(sum(obj)); s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = SOLVE_SEC; s.parameters.num_workers = 4
    return {i: s.Value(gv[i]) for i in range(n)} if s.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE) else {}

def day_run(df, cfg, nrel, rem, rng):
    if len(df) == 0: return pd.DataFrame(columns=COLS)
    df = df.reset_index(drop=True); pl = {}
    if cfg["f1"]:
        q = predQ(df); df["q10"], df["q50"], df["q90"] = q[:, 0], q[:, 1], q[:, 2]; pl = plan(df, cfg, nrel, rem)
    return execute(df, cfg, nrel, pl, rng)

# ---------- 10월 재현(보정)과 검증 ----------
def replay(ds, cfg):
    rng = np.random.default_rng(0)
    r = pd.concat([execute(g.reset_index(drop=True), cfg, 0, {}, rng) for _, g in ds.groupby("Date")])
    pm = [dm(x.gin, x.go) for _, x in r.groupby("day")]
    return dict(road=r.road.mean(), iw=r.iw.mean(), d4=sum(p[0] >= 4 for p in pm), d3=sum(p[0] >= 3 for p in pm))
def observed(ds):
    pm = [dm(x.gin, x.go) for _, x in ds.groupby("Date")]
    return dict(road=ds.ext.mean(), iw=ds.iw.mean(), d4=sum(p[0] >= 4 for p in pm), d3=sum(p[0] >= 3 for p in pm))
BASE = dict(f1=False, f2=False, f3=False, f4=False, pool=False, bays=3, hand=5)
OB = observed(dtr); best = None
for hand in range(0, 13):
    for bays in (2, 3, 4):
        r = replay(dtr, {**BASE, "hand": hand, "bays": bays})
        loss = abs(r["iw"] - OB["iw"]) / OB["iw"] + abs(r["d4"] - OB["d4"]) + 0.5 * abs(r["d3"] - OB["d3"])
        if best is None or loss < best[0]: best = (loss, hand, bays, r)
BASE.update(hand=best[1], bays=best[2])
print(f"\n[② 10월 재현] hand={best[1]}분, bays={best[2]}\n  실제 {({k: round(v, 1) for k, v in OB.items()})}\n  모형 {({k: round(v, 1) for k, v in best[3].items()})}")
print("[② 검증 10/22~31] 실제", {k: round(v, 1) for k, v in observed(dte).items()}, "모형", {k: round(v, 1) for k, v in replay(dte, BASE).items()})

# ---------- 11월 이후 시나리오 ----------
L16 = pd.read_csv(DIR + "16_lookahead_90d.csv"); print("\n16번 컬럼:", list(L16.columns))
PW = L16.pivot_table(index=find(L16, "Week"), columns=find(L16, "Building"), values=find(L16, "Planned_Workers", "Worker", "인원"), aggfunc="sum")
C18 = pd.read_csv(DIR + "18_climate_reference_90d.csv").set_index("Week"); C18["ws"] = pd.to_datetime(C18["Week_Start"]); WEEKS = list(C18.index)
W45 = sorted(PW.index)[0]; print("16번에 없는 주:", [x for x in WEEKS if x not in PW.index])
def IDX(wk, z):
    b = BM.get(z)
    return 1.0 if b not in PW.columns else PW.loc[wk if wk in PW.index else PW.index[-1], b] / PW.loc[W45, b]
OPD = {wk: [x for x in (C18.loc[wk, "ws"] + pd.Timedelta(days=i) for i in range(6)) if x not in HOL] for wk in WEEKS}
LAMD = [x for x in OPD_OCT if x >= "2026-10-14"]
cnt = d[d.Date >= "2026-10-14"].groupby(["Date", "tz"]).size().unstack(fill_value=0).reindex(index=LAMD, columns=ZONES, fill_value=0)
PAR = {z: (cnt[z].mean(), max(cnt[z].var(ddof=1), 1e-9)) for z in ZONES}; TM = {z: d[d.tz == z].reset_index(drop=True) for z in ZONES}
QPOUR = len(REM) / len(OPD_OCT)

def draw(m_, v_, rng):
    if m_ > 0 and v_ > m_ * 1.05: r = m_ * m_ / (v_ - m_); return rng.negative_binomial(r, r / (r + m_))
    return rng.poisson(m_)
def gen_day(day, idxf, rng):
    rows = []
    for z in ZONES:
        k = idxf(z); n = draw(PAR[z][0] * k, PAR[z][1] * k, rng)
        for i in (rng.integers(0, len(TM[z]), n) if len(TM[z]) else []):
            t = TM[z].iloc[i]; rows.append(dict(day=day, tz=z, eq=t.eq, rsv=t.rsv, rsve=t.rsve, arr=t.arr, ext=t.ext, unl=t.unl, exg=t.exg, early=t.early, dn=0))
    return pd.DataFrame(rows, columns=GC)

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
            S.append(dict(day=day, week=wk, blocked=blocked, hold=hold, hot=rng.random() < QHOT, cold=int(row["Avg_Temp_C"] <= 4),
                          rem=list(REM[rng.choice(list(REM))]) if rng.random() < QPOUR else [],
                          trucks=gen_day(day, lambda z, wk=wk: IDX(wk, z), rng)))
    return S

def run_policy(S, cfg, rng):
    carry = None; pend = []; recs = []
    for i, s in enumerate(S):
        tr = s["trucks"]
        if carry is not None and len(carry): tr = pd.concat([tr, carry], ignore_index=True)
        carry = None; flag = cfg["f4"] and rng.random() < (HIT if s["blocked"] else FA); defer = 0
        if flag: pend = [(i if due == i + 1 else due, k) for due, k in pend]          # 불출 앞당김
        nrel = sum(k for due, k in pend if due == i)
        if flag and i + 1 < len(S) and len(tr):                                       # A·C 자재 이연
            mk = tr.tz.isin(ZC) & (tr.dn < 2)
            if mk.any():
                mv = tr[mk].copy(); mv["day"] = S[i + 1]["day"]; mv["dn"] += 1; defer = len(mv); tr = tr[~mk]; carry = mv
        out = day_run(tr, cfg, nrel, s["rem"], rng)
        rec = dict(week=s["week"], n=len(out), road=0., moved=0., iw=0., iw20=0, mx=0, m3=0., m4=0., risk=0.)
        if len(out):
            mxx, m3, m4 = dm(out.gin, out.go)
            rec.update(road=out.road.sum(), moved=out.moved.sum(), iw=out.iw.sum(), iw20=int((out.iw > 20).sum()), mx=mxx, m3=m3, m4=m4, risk=risk_idx(out.gin, out.go))
        pend = [(due, k) for due, k in pend if due > i] + [(i + int(rng.choice(STAY)) + int(s["hot"]), 1) for _ in range(int((out.tz == "D").sum()) if len(out) else 0)]
        area = sum(k for _, k in pend) * LOAD_AREA; ca = B4_AREA if cfg["f2"] else B3_AREA
        rec.update(over=int(area > ca), occ=area / ca, cold=s["cold"],
                   blockAC=int(tr.tz.isin(ZC).sum()) if s["blocked"] and len(tr) else 0, fa_defer=0 if s["blocked"] else defer,
                   hold0=s["hold"], hold1=s["hold"] * (1 - SW) if (flag and s["blocked"]) else s["hold"])
        recs.append(rec)
    return pd.DataFrame(recs)

def agg(R):
    n = max(R.n.sum(), 1); road, iw = R.road.sum() / n, R.iw.sum() / n
    return dict(trucks=n, road=road, moved=R.moved.sum() / n, iw=iw, iw20=R.iw20.sum(), d3=(R.mx >= 3).sum(), d4=(R.mx >= 4).sum(),
                m3=R.m3.sum(), m4=R.m4.sum(), risk=R.risk.sum(), over=R.over.sum(), occ=R.occ.max(), blockAC=R.blockAC.sum(),
                fa_defer=R.fa_defer.sum(), hold0=R.hold0.sum(), hold1=R.hold1.sum(), mat=n * MAT * (road + iw) / REF, cold=R.cold.sum())

FULL = {**BASE, "f1": True, "f2": True, "f3": True, "f4": True, "pool": True, "bays": BASE["bays"] + 1}
CFGS = {"W0 10월 방식": BASE, "B-Flow 전체": FULL, "전체-①": {**FULL, "f1": False},
        "전체-②": {**FULL, "f2": False, "pool": False, "bays": BASE["bays"]},
        "전체-③": {**FULL, "f3": False}, "전체-④": {**FULL, "f4": False}}
WIN = set(C18.index[C18["ws"] >= pd.Timestamp("2026-12-07")])
PER = {"90일 전체": lambda R: R, "동절기 전(~W49)": lambda R: R[~R.week.isin(WIN)], "동절기(W50~)": lambda R: R[R.week.isin(WIN)]}
res = {p: {k: [] for k in CFGS} for p in PER}
for sd in range(N_RUN):
    S = make_scen(sd)
    for nm, cfg in CFGS.items():
        R = run_policy(S, cfg, np.random.default_rng(sd))
        for p, f in PER.items(): res[p][nm].append(agg(f(R)))
MET = ["trucks", "road", "moved", "iw", "iw20", "d3", "d4", "m3", "m4", "risk", "over", "occ", "blockAC", "fa_defer", "hold0", "hold1", "mat", "cold"]
ci = lambda x: f"{np.mean(x):.1f} [{np.percentile(x, 5):.1f}~{np.percentile(x, 95):.1f}]"
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
for p in PER:
    print(f"\n[④ {p}] 평균 [5~95%]")
    print(pd.DataFrame({k: {m: ci([a[m] for a in v]) for m in MET} for k, v in res[p].items()}).T.to_string())
print("\n[쌍비교] W0 → B-Flow 전체 감소율 % (전=0이면 미산출)")
for p in PER:
    o = {}
    for m, m1 in [("road", "road"), ("iw", "iw"), ("iw20", "iw20"), ("d3", "d3"), ("d4", "d4"), ("m3", "m3"), ("m4", "m4"),
                  ("risk", "risk"), ("over", "over"), ("blockAC", "blockAC"), ("hold0", "hold1"), ("mat", "mat")]:
        v = [(a0[m] - a1[m1]) / a0[m] * 100 for a0, a1 in zip(res[p]["W0 10월 방식"], res[p]["B-Flow 전체"]) if a0[m] > 0]
        o[m] = ci(v) if v else "미산출"
    print(p, o)
