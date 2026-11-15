"""KPI 2 보고: B2 하역 대기, B3(전)/B4(후) 적치 체류·점유를 11~1월 주별·월별로 정리.
실행: python3 kpi2_report.py > out/kpi2_report.txt   (먼저 run_kpi2.sh로 kpi2_*_main.csv 등을 만들어 둘 것)
정의
 · B2 대기(분/대) = 장외 대기(road: 안내 도착~입문) + 입문→하역시작(iw). 10월 실측 도착→입문 12.5 + 입문→하역시작 9.2 = 21.7분과 같은 정의.
 · B2 체류(분/대)  = 입문→출문(go - gin). 10월 실측 55.6분.
 · 적치 체류(작업일) = D자재가 하역되어 적치된 날부터 불출되는 날까지. 전: B3(80㎡, 4대분), 후: B4(200㎡, 10대분); 면적은 가정(A10). ※ 체류일은 시뮬레이션 가정(assumptions A09, 평균 1.25일 + 화기 승인 지연)에서 나온 값.
 · 적치 점유율(%) = 그날 적치 물량(대) × 20㎡ ÷ 구역 면적. 100% 초과일 = 면적 초과(하역장 점유 증가 위험, 제약 C08)."""
import glob, os
import numpy as np, pandas as pd, warnings
import assumptions as ASM
from kpi_spec import header
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
TAG = os.environ.get("TAG", "main"); CF = ["W0 10월 방식", "B-Flow 전체"]; NM = {"W0 10월 방식": "전(B2→B3)", "B-Flow 전체": "후(B2→B4)"}
OCT = dict(road=12.5, iw=9.2, total=21.7, b2stay=55.6)       # kpi2_baseline.py 실측(05번 80대)
d = pd.read_csv(f"out/kpi2_days_{TAG}.csv", parse_dates=["day"]); t = pd.read_csv(f"out/kpi2_trucks_{TAG}.csv", parse_dates=["day"]); r = pd.read_csv(f"out/kpi2_release_{TAG}.csv", parse_dates=["day"])
for x in (d, t, r): x["month"] = x.day.dt.strftime("%Y-%m")
t["wait"] = t.road + t.iw; t["stay"] = t.go - t.gin
WEEKS = sorted(d.week.unique(), key=lambda w: d[d.week == w].day.min()); MONTHS = sorted(d.month.unique())
w0 = {w: d[d.week == w].day.min().strftime("%m/%d") for w in WEEKS}; NRUN = d.run.nunique(); ND = d[(d.run == 0) & (d.config == CF[0])].shape[0]
print(header("KPI2"))
print(f"== 자료: 시나리오 {NRUN}개 × 작업일 {ND}일, 방식 {CF}, 태그 {TAG}")

def per(df, key, col, how="mean"):                # run × 기간 값을 만든 뒤 run 간 평균·10%·90%
    g = df.groupby(["config", "run", key])[col]; v = (g.mean() if how == "mean" else g.sum() if how == "sum" else g.max()).reset_index()
    return v.groupby(["config", key])[col].agg(m="mean", p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9)).reset_index()
def wtd(df, key, num, den="n"):                    # 트럭 수 가중 평균(run 내 합 ÷ 트럭 수)
    g = df.groupby(["config", "run", key]).apply(lambda x: pd.Series({"v": x[num].sum() / max(x[den].sum(), 1)})).reset_index()
    return g.groupby(["config", key]).v.agg(m="mean", p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9)).reset_index()
t["n"] = 1
rows = []
def add(metric, unit, key, P):
    for _, x in P.iterrows(): rows.append(dict(지표=metric, 방식=NM[x.config], 단위=unit, 기간=x[key], 평균=x.m, p10=x.p10, p90=x.p90))
for key, lab in (("week", "주"), ("month", "월")):
    add("B2 대기(분/대)", lab, key, wtd(t, key, "wait")); add("B2 장외대기(분/대)", lab, key, wtd(t, key, "road")); add("B2 입문후대기(분/대)", lab, key, wtd(t, key, "iw"))
    add("B2 체류(분/대)", lab, key, wtd(t, key, "stay")); add("적치 체류(작업일)", lab, key, per(r, key, "dwell"))
    add("적치 점유율(%)", lab, key, per(d.assign(occ=d.occ * 100), key, "occ")); add("적치 점유율 최대(%)", lab, key, per(d.assign(occ=d.occ * 100), key, "occ", "max"))
    add("면적 초과일(일)", lab, key, per(d, key, "over", "sum"))
F = pd.DataFrame(rows); F["주시작"] = F.apply(lambda x: w0.get(x.기간, ""), axis=1); F.round(2).to_csv(f"out/kpi2_forecast_{TAG}.csv", index=False)

def show(metric, lab, f="{:5.1f}"):
    x = F[(F.지표 == metric) & (F.단위 == lab)].copy(); x["값"] = x.apply(lambda y: (f + " [" + f + "~" + f + "]").format(y.평균, y.p10, y.p90), axis=1)
    p = x.pivot(index="기간", columns="방식", values="값")[[NM[c] for c in CF]]
    if lab == "주": p.insert(0, "주시작", [w0[k] for k in p.index])
    return p
for m in ("B2 대기(분/대)", "B2 체류(분/대)", "적치 체류(작업일)", "적치 점유율(%)", "면적 초과일(일)"):
    print(f"\n== [{m}] 주별 평균 [시나리오 간 10~90%]"); print(show(m, "주").to_string()); print(f"\n[{m}] 월별"); print(show(m, "월").to_string())

# ---------------- 76작업일 전체 요약 + 쌍비교 ----------------
print("\n== [76작업일 전체 요약] (같은 시나리오끼리 쌍비교, 감소 = 전 − 후)")
def tot(df, f): return df.groupby(["config", "run"]).apply(f).unstack(0)
S = {"B2 대기 평균(분/대)": tot(t, lambda x: x.wait.mean()), "  └ 장외대기": tot(t, lambda x: x.road.mean()), "  └ 입문→하역시작": tot(t, lambda x: x.iw.mean()),
     "B2 체류 평균(분/대)": tot(t, lambda x: x.stay.mean()), "B2 대기 30분 초과 트럭(대)": tot(t, lambda x: (x.wait > 30).sum()), "B2 대기 90% 값(분)": tot(t, lambda x: x.wait.quantile(.9)),
     "안내로 도착 늦춘 시간(분/대, 전은 0)": tot(t, lambda x: x.moved.mean()), "적치 체류 평균(작업일)": tot(r, lambda x: x.dwell.mean()), "적치 체류 3일 이상 비율(%)": tot(r, lambda x: 100 * (x.dwell >= 3).mean()),
     "적치 점유율 평균(%)": tot(d, lambda x: 100 * x.occ.mean()), "적치 점유율 최대(%)": tot(d, lambda x: 100 * x.occ.max()), "면적 초과일(일)": tot(d, lambda x: x.over.sum()),
     "적치 물량 최대(대)": tot(d, lambda x: x.loads.max())}
out = []
for k, v in S.items():
    a, b = v[CF[0]], v[CF[1]]; df_ = a - b
    out.append({"지표": k, "전": f"{a.mean():.2f} [{a.quantile(.1):.2f}~{a.quantile(.9):.2f}]", "후": f"{b.mean():.2f} [{b.quantile(.1):.2f}~{b.quantile(.9):.2f}]",
                "감소(전−후)": f"{df_.mean():.2f} [{df_.quantile(.1):.2f}~{df_.quantile(.9):.2f}]", "후가 더 나쁜 시나리오": f"{int((df_ < 0).sum())}/{len(df_)}"})
O = pd.DataFrame(out); print(O.to_string(index=False)); O.to_csv(f"out/kpi2_summary_{TAG}.csv", index=False)
w0w, bfw = S["B2 대기 평균(분/대)"][CF[0]].mean(), S["B2 대기 평균(분/대)"][CF[1]].mean()
print(f"\n※ 재현 확인: 전(10월 방식) 시뮬레이션 B2 장외대기 {S['  └ 장외대기'][CF[0]].mean():.1f}분, 입문→하역시작 {S['  └ 입문→하역시작'][CF[0]].mean():.1f}분, 합 {w0w:.1f}분 vs 10월 실측 {OCT['road']}+{OCT['iw']}={OCT['total']}분, B2 체류 {S['B2 체류 평균(분/대)'][CF[0]].mean():.1f}분 vs 실측 {OCT['b2stay']}분")

# 목적지(자재 종류)별 B2 대기와 하역 시점 동시 체류
print("\n== [B2 대기: 목적지 구역별 평균(분/대)]"); z = t.groupby(["config", "run", "tz"]).wait.mean().reset_index().groupby(["config", "tz"]).wait.mean().unstack(0)[CF].round(1)
z.columns = [NM[c] for c in CF]; print(z.to_string()); z.to_csv(f"out/kpi2_b2_by_target_{TAG}.csv")
print("\n== [적치 체류 분포: 전체 불출 건 비율(%)]"); dd = r.groupby("config").dwell.value_counts(normalize=True).unstack(0).fillna(0).mul(100).round(1)[CF]; dd.columns = [NM[c] for c in CF]; print(dd.to_string())
fl = d.groupby("config").flag.sum() / NRUN; print(f"\n[후 방식의 불출 앞당김] 앞당김 발동일(시나리오 평균): {fl[CF[1]]:.1f}일/{ND}일 → 그날 다음날 불출분이 당일로 당겨짐. 전 방식은 0일")
cen = d[d.config == CF[0]].groupby("run").loads.last().mean(); print(f"[참고] 예측 구간 마지막 날 적치 중인 물량 평균 {cen:.1f}대 — 불출 전이라 체류 통계에서 빠짐(우측 절단), 영향은 미미")

# ---------------- 가정 민감도 ----------------
print("\n== [체류일 가정 민감도] (각 30개 시나리오, 시드는 본 실행 앞 30개와 같음)")
SR = []
for tg, nm in (("main", "기본 v2 (1·1·1·2일, 본 실행 앞 30개)"), ("sens_stay_v1", "v1 체류 1·2·3일(10월 재생에서 B3 용량 모순)"), ("sens_stay_long", "길게 2·3·4일"),
               ("sens_beta_lo", "반입 탄력성 0.33(95% 하한)"), ("sens_beta_hi", "반입 탄력성 1.27(95% 상한)")):
    if not os.path.exists(f"out/kpi2_days_{tg}.csv"): continue
    dd_ = pd.read_csv(f"out/kpi2_days_{tg}.csv"); rr_ = pd.read_csv(f"out/kpi2_release_{tg}.csv"); tt_ = pd.read_csv(f"out/kpi2_trucks_{tg}.csv"); dd_ = dd_[dd_.run < 30]; rr_ = rr_[rr_.run < 30]; tt_ = tt_[tt_.run < 30]
    for c in CF:
        x = rr_[rr_.config == c]; y = dd_[dd_.config == c]; z = tt_[tt_.config == c]
        SR.append({"가정": nm, "방식": NM[c], "B2 대기(분/대)": round((z.road + z.iw).mean(), 1), "적치 체류 평균(작업일)": round(x.dwell.mean(), 2), "점유율 평균(%)": round(100 * y.occ.mean(), 0), "점유율 최대 평균(%)": round(100 * y.groupby("run").occ.max().mean(), 0),
                   "면적 초과일(76일 중)": round(y.groupby("run").over.sum().mean(), 1)})
if SR: SR = pd.DataFrame(SR); print(SR.to_string(index=False)); SR.to_csv("out/kpi2_sens_stay.csv", index=False)

# ---------------- 그림 ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fn = [f.name for f in fm.fontManager.ttflist]
for nm in ("NanumGothic", "NanumSquare", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if nm in fn: plt.rcParams["font.family"] = nm; break
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(4, 1, figsize=(11, 16)); x = np.arange(len(WEEKS)); lab = [w0[w] for w in WEEKS]; col = {NM[CF[0]]: "tab:red", NM[CF[1]]: "tab:blue"}
def panel(a, metric, title, yl, hline=None, hl_label=None, ylim0=True):
    for c in CF:
        s = F[(F.지표 == metric) & (F.단위 == "주") & (F.방식 == NM[c])].set_index("기간").loc[WEEKS]
        a.plot(x, s.평균, "o-", color=col[NM[c]], label=NM[c]); a.fill_between(x, s.p10, s.p90, color=col[NM[c]], alpha=.15)
    if hline is not None: a.axhline(hline, color="gray", ls="--", label=hl_label)
    a.set_xticks(x); a.set_xticklabels(lab, rotation=60, fontsize=8); a.set_title(title); a.set_ylabel(yl); a.grid(alpha=.3); a.legend(fontsize=8)
    if ylim0: a.set_ylim(bottom=0)
panel(ax[0], "B2 대기(분/대)", "B2 하역장: 트럭 1대당 대기 = 장외대기 + 입문 후 하역 시작까지 (평균, 음영 = 시나리오 10~90%)", "분", OCT["total"], f"10월 실측 {OCT['total']}분")
panel(ax[1], "B2 체류(분/대)", "B2 하역장: 트럭 1대당 체류 (입문→출문)", "분", OCT["b2stay"], f"10월 실측 {OCT['b2stay']}분")
panel(ax[2], "적치 체류(작업일)", "적치 구역 체류: 하역~불출 (전: B3, 후: B4) — 체류일은 가정(1·1·1·2일)에서 나온 값", "작업일")
panel(ax[3], "적치 점유율(%)", "적치 점유율 (전: B3 80㎡, 후: B4 200㎡). 100% 초과 = 면적 초과", "%", 100, "면적 한도")
fig.tight_layout(); fig.savefig(f"out/kpi2_forecast_{TAG}.png", dpi=130); print(f"\n저장: out/kpi2_forecast_{TAG}.csv|png, out/kpi2_summary_{TAG}.csv, out/kpi2_b2_by_target_{TAG}.csv, out/kpi2_sens_stay.csv")
