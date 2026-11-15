"""CP-SAT INFEASIBLE 원인 진단 (검사용). 실행: python3 diag_infeasible.py
bflow_sim.py의 시나리오·계획 입력을 그대로 기록한 뒤, 제약 묶음을 하나씩 빼고 다시 풀어 원인을 찾는다."""
import os, io, contextlib, collections
os.environ["VAL_REP"] = "1"
src = open("bflow_sim.py").read().split("\nFULL = {")[0]
ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "bflow_sim.py", "exec"), ns)
np, pd, cp_model = ns["np"], ns["pd"], ns["cp_model"]
GATE, END, B4_MOVE, REL, CAP = ns["GATE"], ns["END"], ns["B4_MOVE"], ns["REL"], ns["CAP"]
REC = []; orig = ns["plan"]
def rec_plan(df, cfg, nrel, rem):
    pl = orig(df, cfg, nrel, rem); REC.append((df.copy(), dict(cfg), nrel, list(rem), bool(pl))); return pl
ns["plan"] = rec_plan
FULL = {**ns["BASE"], "name": "B-Flow 전체", "f1": True, "f2": True, "f3": True, "f4": True, "pool": True, "bays": ns["BASE"]["bays"] + 1}
for sd in range(3): ns["run_policy"](ns["make_scen"](sd), FULL)

def solve(df, cfg, nrel, rem, gate=True, remg=True, gwid=True, bays=True, fork=True, rel=True, cap=True, win="own"):
    n = len(df); m = cp_model.CpModel(); G, U, F, D = [], [], [], []
    for i, r in enumerate(df.itertuples()):
        ue, ex, hs = int(np.ceil(r.unl)), int(np.ceil(r.exg)), int(round(r.hand))
        if win == "c01":
            b0, b1 = (540, 660) if r.rsv < 720 else (840, 960); lo0, hi0 = min(int(r.rsv), b0), max(int(r.rsve), b1)
        else: lo0, hi0 = int(r.rsv), int(max(r.rsve, r.rsv + 30))
        hi = min(hi0, END - ue - hs - 1); lo = min(lo0, hi)
        g = m.NewIntVar(lo, hi, ""); u = m.NewIntVar(lo + hs, END - ue, ""); m.Add(u >= g + hs)
        G.append(m.NewFixedSizeIntervalVar(g, GATE + (int(np.clip((r.q90 - r.q10) / 4, 0, 10)) if gwid else 0), ""))
        U.append(m.NewFixedSizeIntervalVar(u, ue, "")); F.append(m.NewFixedSizeIntervalVar(u, ue + (B4_MOVE if r.tz == "D" else 0), ""))
        e = m.NewIntVar(0, 1440, ""); m.Add(e == u + ue + ex); sz = m.NewIntVar(0, 1440, ""); D.append(m.NewIntervalVar(g, sz, e, ""))
    if gate: m.AddNoOverlap(G + ([m.NewFixedSizeIntervalVar(int(t), GATE, "") for t in rem] if remg else []))
    if bays: m.AddCumulative(U, [1] * n, cfg["bays"])
    if fork: m.AddCumulative(F + ([m.NewFixedSizeIntervalVar(480, REL, "") for _ in range(nrel)] if rel else []), [1] * (n + (nrel if rel else 0)), 3)
    if cap: m.AddCumulative(D, [1] * n, CAP)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = 2.0; s.parameters.num_workers = 1
    return s.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

bad = [x for x in REC if not x[4]]; print(f"계획을 세운 날 {len(REC)}일 중 INFEASIBLE {len(bad)}일 (B-Flow 전체, 시나리오 3개)")
TESTS = {"불출 작업 고정(08:00) 제거": dict(rel=False), "동시 체류 상한 제거": dict(cap=False), "레미콘 게이트 구간 제거": dict(remg=False),
         "게이트 슬롯 폭 가산 제거(3분 고정)": dict(gwid=False), "게이트 제약 전체 제거": dict(gate=False), "지게차 제약 제거": dict(fork=False),
         "하역 자리 제약 제거": dict(bays=False), "예약 이동 범위를 C01 블록으로 확대": dict(win="c01")}
fix = collections.Counter(); combo = collections.Counter(); none = 0
for df, cfg, nrel, rem, _ in bad:
    ok = [nm for nm, kw in TESTS.items() if solve(df, cfg, nrel, rem, **kw)]
    for nm in ok: fix[nm] += 1
    combo[" / ".join(ok) if ok else "(단독 완화로는 해결 안 됨)"] += 1
print("\n[한 가지만 풀었을 때 풀리는 날 수]"); [print(f"  {k}: {v}") for k, v in fix.most_common()]
print("\n[날짜별 해결 조합]"); [print(f"  {v:3d}일  {k}") for k, v in combo.most_common()]
print("\n[특성] 불출 작업 수(nrel)별 INFEASIBLE 비율:")
t = pd.DataFrame([dict(nrel=x[2], n=len(x[0]), pour=int(len(x[3]) > 0), bad=int(not x[4]), carry=int((x[0].dn > 0).sum())) for x in REC])
print(t.groupby("nrel").bad.agg(["count", "sum", "mean"]).round(2).to_string())
print("트럭 수별:"); print(t.groupby("n").bad.agg(["count", "sum", "mean"]).round(2).to_string())
print("타설일 여부별:"); print(t.groupby("pour").bad.agg(["count", "sum", "mean"]).round(2).to_string())
print("이연 차량 포함 여부별:"); print(t.groupby(t.carry > 0).bad.agg(["count", "sum", "mean"]).round(2).to_string())
w = pd.concat([x[0] for x in REC]); print("\n게이트 슬롯 폭(분) 분포:", (GATE + np.clip((w.q90 - w.q10) / 4, 0, 10).astype(int)).value_counts().sort_index().to_dict())
r0 = sorted(ns["REM"]["2026-10-19"]); print("레미콘 입문 간격(분, 10/19):", [int(b - a) for a, b in zip(r0, r0[1:])], "→ 빈 틈 = 간격 − 3분")
print("차량 1대 최소 체류(준비+하역+출문, 분) 분위수:", (w.hand.round() + np.ceil(w.unl) + np.ceil(w.exg)).quantile([.1, .5, .9]).tolist(), "| 자기 예약 창 길이(분):", (w.rsve - w.rsv).quantile([.1, .5, .9]).tolist())
both = sum(1 for df, cfg, nrel, rem, _ in bad if solve(df, cfg, nrel, rem, rel=False, gwid=False, win="c01")); print(f"\n세 가지(불출 고정 제거 + 슬롯 폭 3분 + C01 블록)를 함께 풀면: {both}/{len(bad)}일 해결")
both2 = sum(1 for df, cfg, nrel, rem, _ in bad if solve(df, cfg, nrel, rem, rel=False, gwid=False)); print(f"자기 예약 창을 유지하고 두 가지(불출 고정 제거 + 슬롯 폭 3분)만 풀면: {both2}/{len(bad)}일 해결")
