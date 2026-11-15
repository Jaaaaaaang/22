"""KPI 2 예측용 시뮬레이션 실행: B2 대기, B3(전)/B4(후) 적치 체류·불출을 11~1월(76작업일) 일별로 기록한다.
bflow_sim.py는 고치지 않고, 그 파일의 앞부분(자료 읽기·수요 생성·하역장 모형·방식 정의)을 그대로 불러온 뒤
run_policy만 '추적 항목을 더한 사본'으로 바꿔 쓴다. 기존 지표는 out/runs_main.csv와 일치해야 한다(kpi2_check.py로 확인).
실행: N_RUN=100 TAG=main python3 kpi2_forecast.py > out/kpi2_run_main.txt
환경변수: N_RUN, TAG, STAY_SET(적치 체류일 후보, 기본 1,2,3 = 시뮬레이션 기본 가정), VAL_REP(수요 생성 검증 반복, 기본 2로 낮춤)
출력: out/kpi2_days_{TAG}.csv(일별), out/kpi2_trucks_{TAG}.csv(트럭별), out/kpi2_release_{TAG}.csv(불출별 체류), out/kpi2_agg_{TAG}.csv(검증용 집계)"""
import os, sys, time
os.environ.setdefault("VAL_REP", "2")
src = open("bflow_sim.py", encoding="utf-8").read()
cut = src.index("res = {p: {k: [] for k in CFGS} for p in PER}")
exec(compile(src[:cut], "bflow_sim.py", "exec"), globals())          # 위 문자열 앞까지(자료·모형·방식 정의)만 실행
STAY = tuple(int(x) for x in E("STAY_SET", "1,2,3").split(","))     # gen_day가 전역 STAY를 읽는다. 난수 개수는 같아 시나리오(트럭·날씨)는 그대로
print(f"\n[KPI2] 체류일 후보 STAY={STAY}, 적치 면적 B3 {B3_AREA}㎡(={B3_AREA // LOAD_AREA}대) / B4 {B4_AREA}㎡(={B4_AREA // LOAD_AREA}대)")

def run_policy2(S, cfg, run, cname):
    """bflow_sim.run_policy와 같은 계산. 추가: 불출 항목에 입고 일(born)을 붙여 체류일을 기록, 트럭별 대기 기록."""
    carry = None; pend = []; recs = []; trk = []; dw = []
    for i, s in enumerate(S):
        tr = s["trucks"]
        if carry is not None and len(carry): tr = pd.concat([tr, carry], ignore_index=True)
        carry = None; flag = cfg["f4"] and s["uf"] < (HIT if s["blocked"] else FA); defer = 0
        if flag: pend = [(i if due == i + 1 else due, k, b) for due, k, b in pend]
        nrel = sum(k for due, k, b in pend if due == i)
        dw += [(run, cname, s["day"], s["week"], i - b) for due, k, b in pend if due == i]               # 오늘 불출되는 물량의 체류 작업일
        if flag and i + 1 < len(S) and len(tr):
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
            tt = out[["tz", "road", "moved", "iw", "held", "gin", "go"]].copy(); tt["run"], tt["config"], tt["day"], tt["week"] = run, cname, s["day"], s["week"]; trk.append(tt)
        nd_ = tr[tr.tz == "D"] if len(tr) else tr
        new = [(i + int(st) + int(s["hot"]), 1, i) for st in (nd_.stay.values if len(nd_) else [])]
        pend = [(due, k, b) for due, k, b in pend if due > i] + new
        loads = sum(k for _, k, _ in pend); ca = B4_AREA if cfg["f2"] else B3_AREA
        rec.update(loads=loads, over=int(loads * LOAD_AREA > ca), occ=loads * LOAD_AREA / ca, cold=s["cold"],
                   blockAC=int(tr.tz.isin(ZC).sum()) if s["blocked"] and len(tr) else 0, fa_defer=0 if s["blocked"] else defer,
                   hold0=s["hold"], hold1=s["hold"] * (1 - SW) if (flag and s["blocked"]) else s["hold"],
                   nrel=nrel, new_d=len(new), flag=int(flag), hot=int(s["hot"]), day=s["day"], run=run, config=cname,
                   cap_loads=ca // LOAD_AREA, n_w30=int(((out.road + out.iw) > 30).sum()) if len(out) else 0)
        recs.append(rec)
    return pd.DataFrame(recs), (pd.concat(trk, ignore_index=True) if trk else pd.DataFrame()), pd.DataFrame(dw, columns=["run", "config", "day", "week", "dwell"])

DAYS_, TRK_, REL_, AGG_ = [], [], [], []
def dump():
    pd.concat(DAYS_, ignore_index=True).to_csv(f"out/kpi2_days_{TAG}.csv", index=False)
    pd.concat(TRK_, ignore_index=True).to_csv(f"out/kpi2_trucks_{TAG}.csv", index=False)
    pd.concat(REL_, ignore_index=True).to_csv(f"out/kpi2_release_{TAG}.csv", index=False)
    pd.DataFrame(AGG_).to_csv(f"out/kpi2_agg_{TAG}.csv", index=False)
t0 = time.time()
for sd in range(N_RUN):
    S = make_scen(sd)
    for nm, cfg in (("W0 10월 방식", BASE), ("B-Flow 전체", FULL)):
        R, TR_, DW_ = run_policy2(S, cfg, sd, nm)
        DAYS_.append(R); TRK_.append(TR_); REL_.append(DW_); AGG_.append(dict(run=sd, config=nm, **{a: float(b) for a, b in agg(R).items()}))
    if (sd + 1) % 10 == 0 or sd + 1 == N_RUN:
        dump(); print(f"  시나리오 {sd + 1}/{N_RUN} 완료 ({time.time() - t0:.0f}초)", flush=True)
print("CP-SAT 풀이 상태:", STAT); print("완료")
