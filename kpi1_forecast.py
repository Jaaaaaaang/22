"""KPI 1 예측 (11월~1월, 2026-W45 ~ 2027-W04, 76작업일): 적치장·통로의 지적(09)·아차사고(10) 발생빈도와 조치 소요.
실행: python3 kpi1_forecast.py        (결과: out/kpi1_forecast_*.csv|png|json, 표준출력은 out/kpi1_forecast.txt로 저장 권장)
환경변수: N_SIM, START(발생률 추정 시작일), QUEUE_EL(B1 대기·분리 지적의 장외대기 탄력성), WINTER_UP(결빙·적설 주의 통로 시설 지적 증가율, 기본 0), TAG, SEED

[범위·방법 — 핵심만]
 · 범위: 09번 중 적치·통로(가+나) 지적, 10번 중 적치 관련 아차사고(가). 10월 기준값(kpi1_baseline.py)과 같은 분류.
 · '전'(10월 방식 지속) = 10월 발생률(세부구역×유형 칸별 감마-포아송)을 11~1월 작업일에 적용.
   노출(작업자 수)은 16번 계획인원을 W45=1.0 지수로 반영하되 탄력성 β는 10월 자료 추정값(0.52±0.47)을 그대로 불확실성에 넣는다.
 · '후'(B-Flow) = 자료로 추정 불가(10월에는 '후'가 없음). 닿는 유형(B3 적치장, D2 임시적치)에 감소율 가정 25/50/75% 시나리오,
   B1 대기·분리 유형은 시뮬레이션의 장외대기 증가율(runs_main.csv의 road 후/전)을 그대로 곱함(악화 방향).
 · 조치 소요: '완료 건은 2.2일 안에, 아니면 10/31까지 미완료'라는 10월 패턴을 그대로 연장(3일 내 조치율 + 미조치 잔량)."""
import os, json
import numpy as np, pandas as pd, warnings
import assumptions as ASM
from kpi_spec import header
warnings.filterwarnings("ignore")
E = os.environ.get
D = "data/"; N_SIM = int(E("N_SIM", 4000)); START = E("START", "2026-10-05"); QUEUE_EL = float(E("QUEUE_EL", 1)); WINTER_UP = float(E("WINTER_UP", 0))
TAG = E("TAG", "main"); SEED = int(E("SEED", 20261115)); ASOF = pd.Timestamp("2026-10-31 17:00")
BETA_MU, BETA_SD = ASM.BETA_INC, ASM.BETA_INC_SD   # 노출 탄력성(assumptions A13: kpi1 분석 A,C,D 구역·일 포아송, n=69, p=0.27)
R_SC = ASM.R_SC
HOL = {pd.Timestamp(x) for x in ASM.HOLIDAYS}
BM = ASM.BM
rng = np.random.default_rng(SEED)
print(header("KPI1"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
qs = lambda x, ax=0: (np.mean(x, ax), np.percentile(x, 10, ax), np.percentile(x, 90, ax))

# ---------------- 자료와 분류 (kpi1_baseline.py와 동일 기준) ----------------
i = pd.read_csv(D + "09_inspection_action_log.csv"); n = pd.read_csv(D + "10_near_miss_reports.csv"); w = pd.read_csv(D + "02_daily_work_plan.csv")
for c in ("Issue_DateTime", "Due_DateTime", "Closed_DateTime"): i[c] = pd.to_datetime(i[c], errors="coerce")
n["Occurrence_DateTime"] = pd.to_datetime(n["Occurrence_DateTime"]); i["Date"] = i.Issue_DateTime.dt.strftime("%Y-%m-%d"); n["Date"] = n.Occurrence_DateTime.dt.strftime("%Y-%m-%d")
STACK = {"STACK_HEIGHT", "MEP_STORAGE", "DUCT_STACK", "REBAR_ROUTE", "REBAR_EDGE", "SIGNAL_VIEW", "PANEL_ACCESS", "GATE_LINE"}
ROUTE = {"WATER_POOL", "RAMP_ROUTE", "RAMP_BARRIER", "TEMP_ROUTE", "PED_BARRIER", "QUEUE_SIGN"}
B3, D2, QU = {"STACK_HEIGHT"}, {"MEP_STORAGE", "DUCT_STACK", "PANEL_ACCESS"}, {"QUEUE_SIGN", "PED_BARRIER", "GATE_LINE"}
C_B3, C_D2, C_Q, C_GA, C_NA = "B3 적치장", "D2 임시적치", "B1 대기·분리", "가 기타(닿지 않음)", "나 기타(닿지 않음)"
def icls(code):
    return C_B3 if code in B3 else C_D2 if code in D2 else C_Q if code in QU else C_GA if code in STACK else C_NA
i = i[i.Finding_Code.isin(STACK | ROUTE)].copy(); i["cls"] = i.Finding_Code.map(icls); i["grp"] = np.where(i.Finding_Code.isin(STACK), "가", "나"); i["kind"] = "지적"
NM_STACK = ("자재·통로 간섭", "철근 적치·협착", "신호수 시야제한")
n = n[n.Event_Type.isin(NM_STACK)].copy(); n["cls"] = np.where(n.Event_Type.eq("자재·통로 간섭"), C_D2, C_GA); n["grp"] = "가"; n["kind"] = "아차사고"
ALLD = sorted(w.Date.unique()); DAYS = [d for d in ALLD if d >= START]; ND = len(DAYS)
cell = pd.concat([i[["kind", "Zone", "cls", "grp", "Date"]], n[["kind", "Zone", "cls", "grp", "Date"]]]).query("Date >= @START")
cells = cell.groupby(["kind", "Zone", "cls", "grp"]).size().rename("cnt").reset_index().rename(columns={"Zone": "zone"})
print(f"== 자료: 지적(가+나) {int((i.Date >= START).sum())}건, 아차사고(가) {int((n.Date >= START).sum())}건, 발생률 추정 구간 {START}~{DAYS[-1]} ({ND}작업일), 칸 {len(cells)}개")
print(cells.to_string(index=False))

# ---------------- 11~1월 달력·노출 지수 (bflow_sim.py와 같은 정의) ----------------
L16 = pd.read_csv(D + "16_lookahead_90d.csv"); PW = L16.pivot_table(index="Week", columns="Building", values="Planned_Workers", aggfunc="sum")
C18 = pd.read_csv(D + "18_climate_reference_90d.csv").set_index("Week"); C18["ws"] = pd.to_datetime(C18["Week_Start"]); WEEKS = list(C18.index); W45 = sorted(PW.index)[0]
IDX = lambda wk, z: 1.0 if BM.get(z) not in PW.columns else PW.loc[wk, BM[z]] / PW.loc[W45, BM[z]]
FD = []
for wk in WEEKS:
    ds = [x for x in (C18.loc[wk, "ws"] + pd.Timedelta(days=k) for k in range(6)) if x not in HOL]
    assert len(ds) == C18.loc[wk, "Planned_Work_Days"], wk
    FD += [(x, wk) for x in ds]
T, C, NW = len(FD), len(cells), len(WEEKS)
MONTHS = sorted({x.strftime("%Y-%m") for x, _ in FD})
WI = np.array([WEEKS.index(wk) for _, wk in FD]); MI = np.array([MONTHS.index(x.strftime("%Y-%m")) for x, _ in FD])
WM = np.eye(NW)[WI]; MM = np.eye(len(MONTHS))[MI]                                    # 일 → 주/월 합산 행렬
print(f"\n== 예측 구간: {FD[0][0]:%Y-%m-%d} ~ {FD[-1][0]:%Y-%m-%d}, {T}작업일, {NW}주, 월 {MONTHS}")
print("노출 지수(16번 계획인원, W45=1.0) 범위:", {BM[z]: f"{min(IDX(wk, z) for wk in WEEKS):.2f}~{max(IDX(wk, z) for wk in WEEKS):.2f}" for z in BM})

# ---------------- 백테스트: 10월 안에서 '그때까지 자료로 앞 6작업일 합계' 예측 ----------------
print("\n== [백테스트] 롤링 원점. 그때까지의 일평균(감마-포아송)으로 다음 6작업일 합계의 80% 구간을 만들고 실제와 비교 (노출 지수는 10월분이 없어 미사용)")
def backtest(df, nm, h=6, cmin=8):
    y = df.groupby("Date").size().reindex(ALLD, fill_value=0).values.astype(float); rows = []
    for c in range(cmin, len(ALLD) - 2):
        hh = min(h, len(ALLD) - c); tr = y[:c]; act = y[c:c + hh].sum()
        pred = rng.poisson(rng.gamma(tr.sum() + .5, 1 / c, 20000) * hh); lo, hi = np.percentile(pred, [10, 90])
        rows.append((act, pred.mean(), lo, hi, y[max(0, c - 6):c].mean() * hh))
    r = np.array(rows); cov = np.mean((r[:, 0] >= r[:, 2]) & (r[:, 0] <= r[:, 3]))
    print(f"  {nm}: 원점 {len(r)}개 | 80% 구간 포함률 {cov:.2f} | MAE 모형(누적 평균) {np.abs(r[:, 0] - r[:, 1]).mean():.2f} vs 직전 6일 평균 {np.abs(r[:, 0] - r[:, 4]).mean():.2f} | 실제 평균 {r[:, 0].mean():.2f}, 예측 평균 {r[:, 1].mean():.2f} (편향 {100 * (r[:, 1].mean() / r[:, 0].mean() - 1):+.0f}%)")
    return cov
cov_i = backtest(i, "지적(가+나)"); cov_n = backtest(n, "아차사고(가)")
print("  (원점 13개는 서로 겹쳐 독립 표본이 아니고 구간 검증력이 약함. 지적은 포함률이 1.00으로 구간이 넓고, 예측 평균이 실제보다 높음 = 10월 후반 발생률이 초반보다 낮았다는 뜻)")
wkc = {int(k): int(v) for k, v in i.assign(wk=pd.to_datetime(i.Date).dt.isocalendar().week).groupby("wk").size().items()}
print("  10월 주별 지적:", wkc, "→ 10/5 주 10건에서 마지막 주 9건으로 소폭 감소. 통계적으로 추세라 할 근거는 약하나 B3 지적은 10/16 이후 0건이므로 'START=2026-10-14' 민감도를 함께 본다")

# ---------------- 시뮬레이션 ----------------
beta = np.clip(rng.normal(BETA_MU, BETA_SD, N_SIM), 0, 1.5)
rate = rng.gamma(cells.cnt.values + .5, 1 / ND, size=(N_SIM, C))                      # 작업일당 발생률(칸별), 제프리스 사전분포
IDXM = np.array([[IDX(wk, z) for z in cells.zone] for _, wk in FD])                    # T×C
WF = np.array([min(1.0, (C18.loc[wk, "Freezing_Days"] + C18.loc[wk, "Expected_Snow_Days"]) / 7) for _, wk in FD])
mult = 1 + WINTER_UP * WF[:, None] * (cells.cls == C_NA).values[None, :]
lam = rate[:, None, :] * IDXM[None] ** beta[:, None, None] * mult[None]
before = rng.poisson(lam)                                                                # N×T×C
runs = pd.read_csv("out/runs_main.csv"); rr_all = (lambda x: (x["B-Flow 전체"] / x["W0 10월 방식"]).values)(runs[runs.period == "90일 전체"].pivot(index="run", columns="config", values="road"))
rr = rng.choice(rr_all, N_SIM) ** QUEUE_EL                                              # 시뮬레이션의 장외대기(road) 후/전 비율, 시나리오 쌍으로 추출
qi = np.where(cells.cls == C_Q)[0]; qa = {}
for c in qi:
    thin = rng.binomial(before[:, :, c], np.minimum(rr, 1)[:, None]); add = rng.poisson(lam[:, :, c] * np.maximum(rr - 1, 0)[:, None]); qa[c] = thin + add
def make_after(r):
    a = before.copy()
    for c in range(C):
        if cells.cls[c] in (C_B3, C_D2): a[:, :, c] = rng.binomial(before[:, :, c], 1 - r)
        elif c in qa: a[:, :, c] = qa[c]
    return a
AFT = {k: make_after(r) for k, r in R_SC.items()}
SC = {"전(10월 방식 지속)": before, **{f"후(B-Flow) 감소가정 {k}": v for k, v in AFT.items()}}
sel = {k: (cells.kind == k).values for k in ("지적", "아차사고")}
tot = lambda arr, k: arr[:, :, sel[k]].sum(2)                                            # N×T

print(f"\n== [가정·설정] 노출 탄력성 β~N({BETA_MU},{BETA_SD}) [0,1.5], B1 대기·분리 지적의 장외대기 탄력성 {QUEUE_EL}(시뮬레이션 road 후/전 평균 {rr_all.mean():.2f}, 범위 {rr_all.min():.2f}~{rr_all.max():.2f}), "
      f"결빙·적설 가산 {WINTER_UP}, 시뮬레이션 {N_SIM}회")
rows = []
for kind in ("지적", "아차사고"):
    for sc, arr in SC.items():
        x = tot(arr, kind)
        for lab, M, names in (("주", WM, WEEKS), ("월", MM, MONTHS)):
            m, lo, hi = qs(x @ M)
            for k, nm in enumerate(names): rows.append(dict(대상=kind, 시나리오=sc, 단위=lab, 기간=nm, 평균=m[k], p10=lo[k], p90=hi[k]))
F = pd.DataFrame(rows); wk0 = {wk: C18.loc[wk, "ws"].strftime("%m/%d") for wk in WEEKS}; F["주시작"] = F.apply(lambda r: wk0.get(r.기간, ""), axis=1)
F.round(2).to_csv(f"out/kpi1_forecast_{TAG}.csv", index=False)

def show(kind, lab):
    t = F[(F.대상 == kind) & (F.단위 == lab)].copy(); t["값"] = t.apply(lambda r: f"{r.평균:5.1f} [{r.p10:4.1f}~{r.p90:4.1f}]", axis=1)
    p = t.pivot(index="기간", columns="시나리오", values="값")[list(SC)]; p.columns = ["전", "후:25%", "후:50%", "후:75%"]; return p
nd_w = pd.Series({wk: int((WI == k).sum()) for k, wk in enumerate(WEEKS)}); nd_m = pd.Series({m: int((MI == k).sum()) for k, m in enumerate(MONTHS)})
for kind in ("지적", "아차사고"):
    print(f"\n== [{kind}] 주별 예측 — 평균 [10~90% 구간] (적치·통로 {'가+나' if kind == '지적' else '가'})")
    p = show(kind, "주"); p.insert(0, "작업일", nd_w); p.insert(0, "주시작", [wk0[x] for x in p.index]); print(p.to_string())
    print(f"\n[{kind}] 월별 예측")
    p = show(kind, "월"); p.insert(0, "작업일", nd_m); print(p.to_string())

b_all = tot(before, "지적").sum(1); print(f"\n== [지적 76작업일 합계] 10월 실적 환산(작업일당 {len(i[i.Date >= START]) / ND:.2f}건 × {T}) = {len(i[i.Date >= START]) / ND * T:.0f}건")
print(f"   전: {b_all.mean():.0f}건 [{np.percentile(b_all, 10):.0f}~{np.percentile(b_all, 90):.0f}] = 주(6작업일)당 {b_all.mean() / T * 6:.1f}건, 작업일당 {b_all.mean() / T:.2f}건")
summ = {"전": dict(지적=float(b_all.mean()), p10=float(np.percentile(b_all, 10)), p90=float(np.percentile(b_all, 90)))}
for k, a in AFT.items():
    ta = tot(a, "지적").sum(1); df_ = b_all - ta; pct = 100 * df_ / np.maximum(b_all, 1); better = (df_ > 0).mean()
    print(f"   후 {k}: {ta.mean():.0f}건 [{np.percentile(ta, 10):.0f}~{np.percentile(ta, 90):.0f}] | 감소 {df_.mean():.1f}건 [{np.percentile(df_, 10):.1f}~{np.percentile(df_, 90):.1f}], 감소율 {pct.mean():.0f}% [{np.percentile(pct, 10):.0f}~{np.percentile(pct, 90):.0f}%] | 후가 전보다 적은 시나리오 {100 * better:.0f}%")
    summ[k] = dict(지적=float(ta.mean()), p10=float(np.percentile(ta, 10)), p90=float(np.percentile(ta, 90)), 감소율=float(pct.mean()))
n_all = tot(before, "아차사고").sum(1); print(f"\n[아차사고 76작업일 합계] 전 {n_all.mean():.1f}건 [{np.percentile(n_all, 10):.0f}~{np.percentile(n_all, 90):.0f}]; " +
      "; ".join(f"후 {k} {tot(a, '아차사고').sum(1).mean():.1f}건" for k, a in AFT.items()))
cl = cells.assign(k=lambda x: x.kind + "/" + x.cls).groupby("k").apply(lambda g: pd.Series({"10월 건": int(g.cnt.sum()), "전(76일 평균)": before[:, :, g.index].sum((1, 2)).mean(),
          "후 50%": AFT["중간 50%"][:, :, g.index].sum((1, 2)).mean()}))
print("\n== 유형별(76작업일 합계, 평균)"); print(cl.round(1).to_string())
reach = cells[(cells.kind == "지적") & cells.cls.isin([C_B3, C_D2])].cnt.sum() / cells[cells.kind == "지적"].cnt.sum()
print(f"닿는 유형 비중(10월): 지적 {100 * reach:.0f}% → 감소율 가정 25~75%일 때 지적 총량 감소 상한은 {25 * reach:.0f}~{75 * reach:.0f}% (B1 대기·분리 증가분은 별도 반영)")

# ---------------- 조치 소요 ----------------
ii = i.copy(); ii["closed"] = ii.Closed_DateTime.notna(); ii["age"] = (ASOF - ii.Issue_DateTime).dt.total_seconds() / 86400; ii["dur"] = (ii.Closed_DateTime - ii.Issue_DateTime).dt.total_seconds() / 86400
old = ii[ii.age >= 3]; k3, m3 = int(old.closed.sum()), len(old); dur = ii.dur.dropna().values
dp = ii[ii.Due_DateTime <= ASOF]; ok = int((dp.Closed_DateTime <= dp.Due_DateTime).sum()); open0 = int((~ii.closed).sum()); lateopen = ii[(~ii.closed) & (ii.age >= 3)]
p3 = rng.beta(k3 + .5, m3 - k3 + .5, N_SIM); pon = rng.beta(ok + .5, len(dp) - ok + .5, N_SIM)
medb = np.array([np.median(rng.choice(dur, len(dur))) for _ in range(N_SIM)])
print(f"\n== [조치 소요] 적치·통로 지적 {len(ii)}건: 완료 {int(ii.closed.sum())}건(모두 {dur.max():.1f}일 이내), 10/31 17시 미조치 {open0}건 (그중 발생 3일 이상 {len(lateopen)}건, 이후 완료된 건 0건)")
print(f"   3일 이상 경과 {m3}건 중 완료 {k3}건 → 3일 내 조치율 {100 * k3 / m3:.0f}% | 기한 도래 {len(dp)}건 중 기한 내 완료 {ok}건 → 기한 내 조치율 {100 * ok / len(dp):.0f}%")
print(f"   3일을 넘긴 건이 나중에 닫힐 확률: 관측 0/{len(lateopen)} → 95% 상한 약 {100 * (1 - 0.05 ** (1 / len(lateopen))):.0f}% (이번 예측은 0으로 둠. 기록 누락이면 실제 잔량은 이보다 적음)")
print(f"   완료 건 소요일 중앙값 {np.median(dur):.2f}일(n={len(dur)}), 부트스트랩 10~90%: {np.percentile(medb, 10):.2f}~{np.percentile(medb, 90):.2f}일 → 완료된 건만의 값이므로 미조치가 늘면 의미가 약해짐")
lt_rows = []; last_idx = [int(np.where(WI == k)[0].max()) for k in range(NW)]; BL = {}
for sc, arr in SC.items():
    x = tot(arr, "지적"); closed = rng.binomial(x, p3[:, None]); bl = open0 + np.cumsum(x - closed, 1); BL[sc] = bl
    m, lo, hi = qs(bl[:, last_idx])
    for k, wk in enumerate(WEEKS): lt_rows.append(dict(시나리오=sc, 주=wk, 주시작=wk0[wk], 미조치잔량_평균=m[k], p10=lo[k], p90=hi[k]))
pd.DataFrame(lt_rows).round(1).to_csv(f"out/kpi1_forecast_backlog_{TAG}.csv", index=False)
LT = pd.DataFrame(lt_rows); LT["값"] = LT.apply(lambda r: f"{r.미조치잔량_평균:5.1f} [{r.p10:4.0f}~{r.p90:4.0f}]", axis=1)
p = LT.pivot(index="주", columns="시나리오", values="값")[list(SC)]; p.columns = ["전", "후:25%", "후:50%", "후:75%"]; p.insert(0, "주시작", [wk0[x] for x in p.index])
print("\n   주말 기준 적치·통로 미조치 잔량 '상한' 시나리오 (10월 말 21건에서 시작. 10월처럼 3일을 넘기면 영영 안 닫힌다고 둔 값이므로 예측치가 아니라 '조치 방식을 안 바꿀 때의 상한'):"); print(p.to_string())
print(f"   → 3일 내 조치율 예측 {100 * p3.mean():.0f}% [{100 * np.percentile(p3, 10):.0f}~{100 * np.percentile(p3, 90):.0f}%], 기한 내 조치율 {100 * pon.mean():.0f}% [{100 * np.percentile(pon, 10):.0f}~{100 * np.percentile(pon, 90):.0f}%]; 조치 방식이 같으면 '후'에서도 같음(B-Flow가 조치 속도를 바꾼다는 근거 없음)")
json.dump(dict(TAG=TAG, START=START, N_SIM=N_SIM, QUEUE_EL=QUEUE_EL, WINTER_UP=WINTER_UP, 지적_76일=summ, 아차사고_76일_전=float(n_all.mean()), 아차사고_76일_후={k: float(tot(a, "아차사고").sum(1).mean()) for k, a in AFT.items()},
               p3=float(p3.mean()), p_ontime=float(pon.mean()), 중앙값_소요일=float(np.median(dur)), 잔량_1월말_전=float(BL["전(10월 방식 지속)"][:, -1].mean()),
               잔량_1월말_후50=float(BL["후(B-Flow) 감소가정 중간 50%"][:, -1].mean()), 백테스트_포함률=dict(지적=float(cov_i), 아차사고=float(cov_n))),
          open(f"out/kpi1_forecast_{TAG}.json", "w"), ensure_ascii=False, indent=1)

# ---------------- 그림 ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fn = [f.name for f in fm.fontManager.ttflist]
for nm in ("NanumGothic", "NanumSquare", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if nm in fn: plt.rcParams["font.family"] = nm; break
plt.rcParams["axes.unicode_minus"] = False
OW = sorted(set(pd.to_datetime(i.Date).dt.isocalendar().week)); ow_lab = [f"{(pd.Timestamp('2026-10-05') + pd.Timedelta(weeks=k)):%m/%d}" for k in range(len(OW))]
fig, ax = plt.subplots(3, 1, figsize=(11, 13)); xo = np.arange(len(OW)); xf = np.arange(NW) + len(OW); lab = ow_lab + [wk0[x] for x in WEEKS]
def wkline(a, kind, title):
    a.bar(xo, [ (i if kind == "지적" else n).assign(wk=lambda d: pd.to_datetime(d.Date).dt.isocalendar().week).groupby("wk").size().reindex(OW, fill_value=0).values[k] for k in range(len(OW))], color="gray", label="10월 실적")
    b = tot(before, kind) @ WM; m, lo, hi = qs(b); a.plot(xf, m, "o-", color="tab:red", label="전: 10월 방식 지속"); a.fill_between(xf, lo, hi, color="tab:red", alpha=.15, label="전 10~90%")
    a25 = (tot(AFT["낮음 25%"], kind) @ WM).mean(0); a50 = (tot(AFT["중간 50%"], kind) @ WM).mean(0); a75 = (tot(AFT["높음 75%"], kind) @ WM).mean(0)
    a.plot(xf, a50, "s-", color="tab:blue", label="후: B-Flow(닿는 유형 감소 50% 가정)"); a.fill_between(xf, a75, a25, color="tab:blue", alpha=.2, label="후 감소 25~75% 가정 범위")
    a.axvline(len(OW) - .5, color="k", ls=":"); a.set_xticks(range(len(lab))); a.set_xticklabels(lab, rotation=60, fontsize=8); a.set_title(title); a.set_ylabel("주당 건수"); a.legend(fontsize=8, ncol=2); a.grid(alpha=.3)
wkline(ax[0], "지적", "적치·통로 지적(가+나) 주별 — 10월 실적과 11~1월 예측 (W52·W53은 5작업일)")
wkline(ax[1], "아차사고", "적치 관련 아차사고(가) 주별 — 10월 실적과 11~1월 예측")
for sc, col in zip(SC, ("tab:red", "tab:blue", "tab:blue", "tab:blue")):
    if sc in ("전(10월 방식 지속)", "후(B-Flow) 감소가정 중간 50%"):
        b = BL[sc][:, last_idx]; m, lo, hi = qs(b); ax[2].plot(range(NW), m, "o-", color=col, label=sc); ax[2].fill_between(range(NW), lo, hi, color=col, alpha=.15)
ax[2].set_xticks(range(NW)); ax[2].set_xticklabels([wk0[x] for x in WEEKS], rotation=60, fontsize=8); ax[2].set_title(f"적치·통로 미조치 잔량 — 상한 시나리오 (10월 조치 방식 그대로, 3일 내 조치율 {100 * p3.mean():.0f}%, 3일 넘기면 미종결)"); ax[2].set_ylabel("건"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"out/kpi1_forecast_{TAG}.png", dpi=130); print(f"\n저장: out/kpi1_forecast_{TAG}.csv|json|png, out/kpi1_forecast_backlog_{TAG}.csv")
