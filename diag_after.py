"""적용 후 운영이 현장 제약을 지키는지 확인 (검사용). 실행: python3 diag_after.py
bflow_sim.py 파일은 그대로 두고, 메모리에서만 execute()에 기록 한 줄을 덧붙여 차량별 시각을 모은다.
본 실행과 같은 시나리오(시드 0~9)를 쓴다."""
import os, io, contextlib
os.environ["VAL_REP"] = "1"; NS = int(os.environ.get("DIAG_N", 10))
src = open("bflow_sim.py").read().split("\nFULL = {")[0]
a = "        rows.append((r.day, r.tz, t - a[j], max(0.0, a[j] - r.arr), us - t, t, go, f, held))\n"
b = a.rstrip("\n") + '; LOG.append(dict(cfg=LABEL[0], day=r.day, tz=r.tz, rsv=r.rsv, arr0=r.arr, arr=a[j], gin=t, us=us, ue=ue, go=go, held=held, pl=pl.get(int(j)), comply=int(int(j) in pl and r.u < P)))\n'
assert src.count(a) == 1; src = "LOG = []; LABEL = ['']\n" + src.replace(a, b)
ns = {}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "bflow_sim.py(diag)", "exec"), ns)
np, pd = ns["np"], ns["pd"]; BASE = ns["BASE"]; d = ns["d"]
FULL = {**BASE, "name": "B-Flow 전체", "f1": True, "f2": True, "f3": True, "f4": True, "pool": True, "bays": BASE["bays"] + 1}
ns["LOG"].clear()
for lab, cfg, cap in (("W0 10월 방식", BASE, 2), ("B-Flow 전체(상한 2대)", FULL, 2), ("B-Flow 전체(상한 3대)", FULL, 3)):
    ns["CAP"] = cap; ns["LABEL"][0] = lab
    for sd in range(NS): ns["run_policy"](ns["make_scen"](sd), cfg)
L = pd.DataFrame(ns["LOG"]); L["road"] = L.gin - L.arr; inwin = lambda m: ((m >= 540) & (m <= 660)) | ((m >= 840) & (m <= 960))
print(f"10월 실제(일반 트럭 80건): 하역 종료 최댓값 {int((d.us + d.unl).max()) // 60}:{int((d.us + d.unl).max()) % 60:02d}, 17시 넘긴 하역 {(d.us + d.unl > 1020).sum()}건, "
      f"도착→입문 평균 {d.ext.mean():.1f}분·최대 {d.ext.max():.0f}분, 입문이 C01 창(09~11·14~16시) 밖 {(~inwin(d.gin)).sum()}건")
print(f"\n시나리오 {NS}개(각 76 작업일) 합계\n")
rows = {}
for lab, x in L.groupby("cfg", sort=False):
    p = x[x.pl.notna()]; sh = (p.pl - p.rsv)
    rows[lab] = {
        "트럭 수": len(x),
        "하역 종료가 17:00 이후(건)": int((x.ue > 1020).sum()),
        "  └ 100대당": round(100 * (x.ue > 1020).mean(), 2),
        "하역 시작이 17:00 이후(건)": int((x.us > 1020).sum()),
        "가장 늦은 하역 종료": f"{int(x.ue.max()) // 60}:{int(x.ue.max()) % 60:02d}",
        "도착→입문 평균(분)": round(x.road.mean(), 1),
        "도착→입문 90% 분위(분)": round(x.road.quantile(.9), 1),
        "도착→입문 최대(분)": round(x.road.max(), 0),
        "도착→입문 30분 초과(100대당)": round(100 * (x.road > 30).mean(), 1),
        "도착→입문 60분 초과(100대당)": round(100 * (x.road > 60).mean(), 1),
        "상한으로 게이트 보류(100대당)": round(100 * x.held.mean(), 1),
        "입문이 C01 창 밖(100대당)": round(100 * (~inwin(x.gin)).mean(), 1),
        "계획이 예약을 옮긴 폭 |계획−예약| 평균(분)": round(sh.abs().mean(), 1) if len(p) else "-",
        "  └ 90% 분위(분)": round(sh.abs().quantile(.9), 1) if len(p) else "-",
        "  └ 30분 넘게 옮긴 비율(%)": round(100 * (sh.abs() > 30).mean(), 1) if len(p) else "-",
        "  └ 앞당긴 비율(%)": round(100 * (sh < -0.5).mean(), 1) if len(p) else "-",
        "계획이 없는 차량 비율(%)": round(100 * x.pl.isna().mean(), 1),
    }
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(pd.DataFrame(rows).to_string())
print("\n오전/오후별 17:00 이후 하역 종료(건):")
print(L.assign(blk=np.where(L.rsv < 720, "오전 예약", "오후 예약"), late=(L.ue > 1020).astype(int)).groupby(["cfg", "blk"], sort=False).late.agg(["count", "sum"]).to_string())
L.to_csv("out/diag_after_trucks.csv", index=False)
