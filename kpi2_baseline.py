"""KPI 2 기준값: B구역(B2 하역장 · B3 적치장 · B4 신설 적치구역)에서 자재가 불출되기까지 기다리는 시간.
10월 자료로 '잴 수 있는 것'과 '못 재는 것'을 구분하고, 잴 수 있는 값은 기준값으로 저장한다.
실행: python3 kpi2_baseline.py > out/kpi2_baseline.txt

해석: '불출 대기' = (1) B2: 트럭이 도착해서 하역이 시작될 때까지 / 하역장에 머무는 시간,
      (2) B3·B4: 적치된 자재가 필요 시점에 불출되기까지 쌓여 있는 시간(체류).
      작업팀이 자재를 기다린 시간(02번 Crew_Waiting)과 지게차 대기(07번)는 보조 지표로 함께 본다."""
import glob, os
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
D = "data/"; os.makedirs("out", exist_ok=True)
T = lambda s: pd.to_datetime(s, errors="coerce")
dl = pd.read_csv(D + "05_delivery_log.csv")
for c in ("Reserved_Start", "Reserved_End", "Arrival_Time", "Gate_In_Time", "Unload_Start_Time", "Unload_End_Time", "Gate_Out_Time"): dl[c] = T(dl[c])
g = dl[dl.Truck_ID.str.startswith("TRK")].copy()                      # 대형 자재 트럭 80대(B2 하역). REM 36대는 B1→A4 레미콘이라 제외
mn = lambda a, b: (g[a] - g[b]).dt.total_seconds() / 60
g["도착→입문"] = mn("Gate_In_Time", "Arrival_Time"); g["입문→하역시작"] = mn("Unload_Start_Time", "Gate_In_Time"); g["하역"] = mn("Unload_End_Time", "Unload_Start_Time")
g["하역끝→출문"] = mn("Gate_Out_Time", "Unload_End_Time"); g["B2 체류(입문→출문)"] = mn("Gate_Out_Time", "Gate_In_Time"); g["도착→하역시작(총 대기)"] = mn("Unload_Start_Time", "Arrival_Time")
g["예약대비 도착(+=이른)"] = mn("Reserved_Start", "Arrival_Time")
print(f"== [0] 05번 대형 자재 트럭 {len(g)}대 (결측: 도착 {g.Arrival_Time.isna().sum()}, 입문 {g.Gate_In_Time.isna().sum()}, 하역시작 {g.Unload_Start_Time.isna().sum()}, 출문 {g.Gate_Out_Time.isna().sum()}), 작업일 {g.Date.nunique()}일")

print("\n== [1] B2 하역장: 구간별 시간(분)")
cols = ["도착→입문", "입문→하역시작", "하역", "하역끝→출문", "B2 체류(입문→출문)", "도착→하역시작(총 대기)"]
t = pd.DataFrame({c: [g[c].mean(), g[c].median(), g[c].quantile(.9), g[c].max(), (g[c] > 20).sum()] for c in cols}, index=["평균", "중앙값", "90%", "최대", "20분 초과(대)"]).T
print(t.round(1).to_string())
print("→ '불출 대기'로 볼 수 있는 B2 지표 = 입문→하역시작(지게차·베이 배정 대기) 평균 %.1f분, 도착→하역시작 평균 %.1f분(예약 시작 전 도착분 포함)" % (g["입문→하역시작"].mean(), g["도착→하역시작(총 대기)"].mean()))
big = g[g["입문→하역시작"] > 20]; print("입문→하역시작 20분 초과:", big[["Delivery_ID", "Date", "Unloading_Equipment_ID", "입문→하역시작"]].values.tolist(), "(지게차가 앞 차량 하역 중이던 건)")
def peak(x):
    ev = sorted([(a, 1) for a in x.Gate_In_Time] + [(b, -1) for b in x.Gate_Out_Time]); c = m = 0
    for _, k in ev: c += k; m = max(m, c)
    return m
pk = g.groupby("Date").apply(peak); print(f"일별 B2 최대 동시 체류: 평균 {pk.mean():.2f}대, 3대 이상인 날 {(pk >= 3).sum()}/{len(pk)}일, 4대 이상 {(pk >= 4).sum()}일")
g["wk"] = pd.to_datetime(g.Date).dt.isocalendar().week.astype(int)
wkt = g.groupby("wk").agg(트럭=("Delivery_ID", "size"), 작업일=("Date", "nunique"), 입문_하역시작_평균=("입문→하역시작", "mean"), 도착_입문_평균=("도착→입문", "mean"), 총대기_평균=("도착→하역시작(총 대기)", "mean"), B2체류_평균=("B2 체류(입문→출문)", "mean"))
print("\n10월 주별(트럭 수가 적어 주별 값은 흔들림):"); print(wkt.round(1).to_string())
t.round(2).to_csv("out/kpi2_b2_october.csv"); wkt.round(2).to_csv("out/kpi2_b2_weekly_october.csv")

print("\n== [2] 보조 지표: 지게차 대기(07번), 작업팀 자재 대기(02번)")
e = pd.read_csv(D + "07_equipment_log.csv"); f = e[e.Equipment_ID.str.startswith("FL")]
shift = 540.0
print("지게차 3대 일평균 대기 %.0f분(교대 9시간 중 %.0f%%), 가동 %.0f분" % (f.groupby("Equipment_ID").Waiting_Minutes.mean().mean(), 100 * f.Waiting_Minutes.mean() / shift, f.Operation_Minutes.mean()))
print(f.groupby("Notes").Waiting_Minutes.agg(["count", "sum", "mean"]).round(1).to_string())
w = pd.read_csv(D + "02_daily_work_plan.csv")
print("B2(T008 하역·분류)·B3(T009 적치·불출) 작업팀 Crew_Waiting_Minutes 합계:", w[w.Task_ID.isin(["T008", "T009"])].Crew_Waiting_Minutes.sum(), "분 (23일 모두 0 → 기록상 B구역 작업팀의 대기는 없음)")
mw = w[w.Daily_Remark.eq("자재반입 확인 대기")]
print(f"'자재반입 확인 대기' 사유 {len(mw)}건, 합계 {mw.Crew_Waiting_Minutes.sum()}분, 평균 {mw.Crew_Waiting_Minutes.mean():.1f}분 — 발생처: {mw.groupby('Subzone').size().to_dict()} (B구역이 아니라 사용처 A1·A2의 팀이 기다린 시간)")

print("\n== [3] B3 적치장: 불출 시각 기록이 있는가?")
hit = {}
for fn in sorted(glob.glob(D + "*.csv")):
    if "00_data_dictionary" in fn: continue
    x = pd.read_csv(fn); s = x.astype(str)
    for c in x.columns:
        k = int(s[c].str.contains("불출").sum())
        if k: hit[f"{os.path.basename(fn)}:{c}"] = k
print("'불출'이라는 값이 들어 있는 칸:", hit)
L = pd.read_csv(D + "06_lifting_log.csv")
print("양중대장(06) 출발 세부구역:", L.Source_Subzone.value_counts().to_dict(), "→ B3에서 나간 기록 없음")
b3 = dl[dl.Related_Staging_Subzone == "B3"]
print(f"B3로 들어온 자재: {len(b3)}대(목적지 {b3.Final_Target_Subzone.value_counts().to_dict()}, 품목 {b3.Material_Category.value_counts().to_dict()}), 일평균 {len(b3) / dl.Date.nunique():.2f}대")
print("→ 입고(05번 B3 관련 반입)는 있으나, 어느 자재가 언제 B3를 떠났는지(불출) 기록이 없어 B3 체류시간은 10월 자료로 산출할 수 없다.")
print("   (kpi1_baseline.py와 시뮬레이션은 '체류 1~3작업일, 평균 2일'을 가정으로 넣었다. 이 값은 자료에서 나온 것이 아니다)")
c4 = pd.read_csv(D + "04_camera_object_log.csv"); cb = c4[c4.Subzone.isin(["B2", "B3"])]
print("카메라(04) B3: 사람 평균 %.1f명, 지게차 평균 %.2f대, 트럭 0 — 적치량·체류는 찍히지 않음" % (cb[cb.Subzone == "B3"].People_Count.mean(), cb[cb.Subzone == "B3"].Forklift_Count.mean()))
print("\n== [4] B4: 10월에는 존재하지 않음 (B-Flow 신설 구역). 관련 기록 0건 → 시뮬레이션 값만 가능")

print("\n== [5] 참고: 같은 '적치 → 불출' 구조인 다른 적치장 A3·C2·C3 (양중 불출, 06번 ↔ 05번 하역 종료)")
x = L.merge(dl[["Delivery_ID", "Unload_End_Time", "Related_Staging_Subzone"]], left_on="Related_Delivery_ID", right_on="Delivery_ID", how="left")
for c in ("Planned_Start", "Actual_Start"): x[c] = T(x[c])
x["적치→불출(h)"] = (x.Actual_Start - x.Unload_End_Time).dt.total_seconds() / 3600
hold_del = set(x[x.Status.eq("보류")].Related_Delivery_ID.dropna()); x["보류 연관"] = x.Related_Delivery_ID.isin(hold_del)
ok = x[x.Status.eq("완료") & x["적치→불출(h)"].notna()]
def sm(df, nm): return pd.Series({"구분": nm, "건": len(df), "평균(h)": df["적치→불출(h)"].mean(), "중앙값(h)": df["적치→불출(h)"].median(), "90%(h)": df["적치→불출(h)"].quantile(.9), "최대(h)": df["적치→불출(h)"].max()})
rows = [sm(ok, "전체")] + [sm(ok[ok.Source_Subzone == z], f"출발 {z}") for z in ("A3", "C2", "C3")] + [sm(ok[~ok["보류 연관"]], "풍속 보류와 무관한 건"), sm(ok[ok["보류 연관"]], "풍속 보류 연관 건")]
R = pd.DataFrame(rows).set_index("구분"); print(R.round(2).to_string()); R.round(2).to_csv("out/kpi2_staging_analog_october.csv")
print("→ 보류와 무관하면 중앙값 약 %.1f시간. 보류가 끼면 다음날로 넘어가 약 %.0f시간 이상이 됨 = 적치 체류는 '양중 가능 여부'가 좌우" % (ok[~ok["보류 연관"]]["적치→불출(h)"].median(), ok[ok["보류 연관"]]["적치→불출(h)"].median()))
print("   계획 시작 대비 실제 시작 지연은 모든 양중이 정확히 5분(자료상 고정값)이라 불출 대기의 정보가 없음:", sorted(((x.Actual_Start - x.Planned_Start).dt.total_seconds() / 60).dropna().unique().tolist()))
print("   ※ 이 값은 B3가 아니라 A3·C2·C3(양중 대상)의 값이다. D자재(B3→D2·D3, 지게차·램프 이동)에 그대로 쓸 수 없다.")

print("\n== [6] 요약: 10월 기준값을 낼 수 있는가")
print(pd.DataFrame([["B2", "도착→하역시작 대기, 입문→하역시작 대기, B2 체류", "가능(05번 80대)", f"{g['입문→하역시작'].mean():.1f}분 / {g['도착→하역시작(총 대기)'].mean():.1f}분 / {g['B2 체류(입문→출문)'].mean():.1f}분"],
                    ["B3", "적치 체류(입고→불출)", "불가(불출 기록 없음)", "가정값(평균 2작업일)만 있음"], ["B4", "적치 체류", "불가(10월에 없음)", "시뮬레이션만 가능"]], columns=["구역", "지표", "10월 실측", "값"]).to_string(index=False))
