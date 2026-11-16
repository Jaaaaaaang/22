"""KPI별 '데이터파일·연결키·산식·가정·제외기준' 명세. 보고서 머리글(header)과 KPI_SPEC.md / out/kpi_spec.csv를 만든다.
실행: python3 check_assumptions.py > out/check_assumptions.txt; python3 kpi_spec.py     (KPI_SPEC.md, out/kpi_spec.csv 생성)
kpi1_forecast.py, kpi2_report.py는 `from kpi_spec import header`로 같은 내용을 결과 파일 맨 위에 출력한다."""
import os
import pandas as pd
import assumptions as A

_S = ",".join(map(str, A.STAY))
SPEC = [
    dict(KPI="KPI1", 지표="적치·통로 지적 발생빈도 (건/주·월)",
         데이터파일="09_inspection_action_log.csv(Issue_DateTime, Zone, Subzone, Finding_Code) · 02_daily_work_plan.csv(Date, Building, Planned_Workers: 작업일 목록) · 16_lookahead_90d.csv(Week, Building, Planned_Workers) · 18_climate_reference_90d.csv(Week, Week_Start, Planned_Work_Days) · out/runs_main.csv(시뮬레이션 장외대기 road, B1 유형에만 사용)",
         연결키="09.Issue_DateTime의 날짜 = 02.Date(작업일) | 09.Zone(A·C·D) ↔ 16.Building(1동·포디움·2동) 규칙 BM | 16.Week = 18.Week | 09.Finding_Code → 분류표(STACK/ROUTE, 닿는 유형 B3·D2·B1) | runs_main.csv: run·config(W0/B-Flow 전체)",
         산식="전: N(칸,t) ~ Poisson(λ칸 × IDX(주,구역)^β), λ칸 ~ Gamma(10월 건수+0.5, 1/작업일수)(Jeffreys), IDX = 16번 계획인원(주)/W45, β ~ N(0.52, 0.47) 절단[0,1.5]. 후: B3·D2 닿는 유형은 Binomial(전, 1−r) r=25/50/75%, B1 대기·분리 유형은 Binomial(전, min(rr,1)) + Poisson(λ·max(rr−1,0)), rr = 시뮬레이션 road 후/전(시나리오 쌍 추출). 76작업일을 주·월로 합산, 평균과 10~90% 구간",
         가정="A13 β(지적 탄력성), A23 후 감소율 25/50/75%, A12 W45 기준, A24 휴일, A06·A07(rr을 통해 시뮬레이션 가정 전체 영향), 결빙·적설 가산은 기본 0",
         제외기준="Finding_Code가 적치·통로(가+나)가 아닌 지적('다' 그룹)은 제외 | 아차사고는 KPI1-b | 발생률 추정 시작 START(기본 10/05=전 기간, 민감도 10/14) | 비작업일 발생 0건(확인) | 10월에 닿지 않는 유형은 감소 적용 안 함",
         실측="전: 10월 실측(23작업일) 있음 / 후: 10월에 없음 → 시나리오 범위로만 제시"),
    dict(KPI="KPI1", 지표="적치 관련 아차사고 발생빈도 (건/주·월)",
         데이터파일="10_near_miss_reports.csv(Occurrence_DateTime, Zone, Event_Type) · 02 · 16 · 18 (위와 같음)",
         연결키="10.Occurrence_DateTime의 날짜 = 02.Date | 10.Event_Type → 분류(자재·통로 간섭 = D2 임시적치, 철근 적치·협착·신호수 시야제한 = 닿지 않음) | 나머지 연결은 위와 같음",
         산식="지적과 같은 감마-포아송·노출 지수. 후: D2 임시적치 유형(자재·통로 간섭)에만 감소율 r 적용",
         가정="A13, A23, A12, A24",
         제외기준="Event_Type이 NM_STACK(자재·통로 간섭, 철근 적치·협착, 신호수 시야제한) 외인 건 제외 | 10번에는 조치 완료일시 컬럼이 없어 조치 소요 산출 불가(Action_Status만 있음)",
         실측="전: 10월 실측 있음 / 후: 10월에 없음"),
    dict(KPI="KPI1", 지표="조치 소요(3일 내 조치율, 기한 내 조치율, 미조치 잔량)",
         데이터파일="09_inspection_action_log.csv(Issue_DateTime, Due_DateTime, Closed_DateTime, Status)",
         연결키="09.Inspection_ID(고유, 중복 0건 확인) | 기준 시각 10/31 17:00(과제안내서)",
         산식="3일 내 조치율 p3 ~ Beta(k+0.5, m−k+0.5), m = 발생 후 3일 이상 경과 건, k = 그중 완료 건 | 기한 내 조치율 = 기한 도래 건 중 Closed ≤ Due | 잔량(t) = 10/31 미조치 + Σ(신규 − Binomial(신규, p3)) (3일을 넘기면 영영 안 닫힌다고 둔 '상한')",
         가정="3일 넘긴 건은 이후 닫히지 않음(관측 0/n, 기록 누락이면 실제 잔량은 더 적음) | B-Flow가 조치 속도를 바꾼다는 근거 없음 → 후도 같은 p3",
         제외기준="경과 3일 미만 건은 p3 분모에서 제외(우측 절단) | 소요일 중앙값은 완료 건만(미조치 제외) | 지적(가+나) 외 분류 제외",
         실측="전: 10월 실측 있음 / 후: 조치 방식이 같다는 가정으로 동일값"),
    dict(KPI="KPI2", 지표="B2 대기 시간 (분/대) = 장외대기 + 입문→하역시작",
         데이터파일="05_delivery_log.csv(Truck_ID, Date, Arrival_Time, Gate_In_Time, Unload_Start_Time, Unload_End_Time, Gate_Out_Time, Reserved_Start/End, Final_Target_Zone, Unloading_Equipment_ID) · 15_site_constraints.csv(C01 예약 창, C03 17시) · 16·18 (수요·달력) · 02(구역·일 계획인원, 수요 탄력성) · 17_equipment_registry.csv(지게차 3대) · 07_equipment_log.csv · 04_camera_object_log.csv(B2 Truck_Count: 슬롯 동시 체류 검증) · 13_weather_hourly.csv, 06_lifting_log.csv(풍속·보류) · 11_work_permits.csv(화기) · 01_master_schedule.csv(T018 종료일)",
         연결키="05.Delivery_ID(트럭) | 05.Date = 02.Date | 05.Final_Target_Zone(A·C·D) ↔ 16.Building ↔ 02.Building | 05.Unloading_Equipment_ID = 17.Equipment_ID = 07.Equipment_ID | 06.Related_Delivery_ID = 05.Delivery_ID | 04: Date·Subzone(B2)·Slot_Start(시간) = 05 입·출문 시각으로 계산한 슬롯 | 11.Task_ID = 01.Task_ID(T018)",
         산식="10월 실측: (Gate_In − Arrival) + (Unload_Start − Gate_In) = 12.5 + 9.2 = 21.7분(80대). 모의: 전 = 10월 규칙(구역 담당 지게차, 베이 3, 선착순 게이트) 실행, 후 = ①분위수 회귀로 예약 시작을 C01 창 안에서 CP-SAT 재배정 → 게이트(GATE 분 + 불확실 완충) 겹침 금지, 하역 동시 ≤ 베이, 지게차 용량, 동시 체류 ≤ CAP → ②공용 지게차 풀·B4 → ③상한 → ④강풍 예보 시 A·C 자재 이연. 100 시나리오(날씨·반입·화기 난수), 쌍비교(같은 시나리오 전−후)",
         가정="A01 GATE, A03 LAG, A04 REL, A05 B4_MOVE, A06 P_COMPLY, A07 CAP, A08 베이, A11 BETA_TRK, A12 수요 기준, A14 화기, A15 HIT/FA, A17 PH, A18 수요 기준일, A19 PREF, A20 지게차 비하역 업무, A21 예약 이동 범위, A22 분위수 모형",
         제외기준="REM-* 레미콘 36건은 대기 통계에서 제외(별도 승인 연속타설, C01 예외; 게이트 점유 시각만 반영) | 05번 결측 0건(제외 없음) | C01 창 밖 예약 4건(DL-0001~0004)은 분위수 학습·수요 풀에서 제외하되 10월 재현에는 포함 | 입문 전 지게차 대기로 하역 시작이 늦은 2건은 '입문→하역준비(hand)'를 나머지 중앙값으로 대체 | 수요는 D1 램프가 열린 10/14 이후 16일만 사용",
         실측="전: 10월 실측 있음(05번 80대). 재현 검증: 지게차 배정 80/80, 하역 시작 ±1분 80/80 | 후: 시뮬레이션만"),
    dict(KPI="KPI2", 지표="B2 체류 시간 (분/대) = Gate_Out − Gate_In",
         데이터파일="05_delivery_log.csv(Gate_In_Time, Gate_Out_Time) + 위 B2 대기와 동일",
         연결키="05.Delivery_ID | 04 슬롯 Truck_Count와 같은 정의의 동시 체류(10월 230슬롯 일치 확인)",
         산식="체류 = 대기 이후 하역(Unload_End − Unload_Start, 10월 관측값 재사용) + 출문(Gate_Out − Unload_End). 10월 실측 55.6분",
         가정="B2 대기와 동일(하역·출문 시간은 10월 관측값을 그대로 재사용, 가정 아님)",
         제외기준="REM-* 레미콘 제외 | 하역 시간 결측은 출문시각으로 역산(평균 대입 안 함)",
         실측="전: 10월 실측 있음 / 후: 시뮬레이션만"),
    dict(KPI="KPI2", 지표="B3(전)/B4(후) 적치 체류 (작업일)",
         데이터파일="05_delivery_log.csv(Final_Target_Zone = D, Date: D자재 하역일) · 06_lifting_log.csv(출발 세부구역이 A3·C2·C3뿐 → B3 불출 기록 없음 확인) · 11·01(화기 지연) · 13·06(풍속 보류)",
         연결키="05.Date(하역일) → 불출일 = 하역일 + 체류(가정) | 05.Final_Target_Zone = D | 11.Permit_ID(PTW-H) ↔ 01.Task_ID(T018)",
         산식=f"하역일 i의 D자재 1대마다 체류 s ~ 균등 {{{_S}}}, 불출일 = i + s + (화기 지연 1일, 확률 2/11, T018 종료 예측일까지) → 체류 = 불출 작업일 − 입고 작업일. 후: 강풍 예보 시 다음날 불출분을 당일로 앞당김(불출 앞당김). 앞당김은 불출일만 당김",
         가정="A09 STAY(1·1·1·2, 평균 1.25일), A14 화기, A15 HIT/FA, A04 REL, 불출은 필요 시점에 즉시 가능(06번 보류 외 대기 없음)",
         제외기준="10월 직접 실측은 불가(불출 기록 없음) → 체류 분포는 가정 | 예측 구간 마지막 날 미불출 물량은 체류 통계에서 제외(우측 절단) | A·C 자재는 양중 대상이라 B3·B4 체류 대상에서 제외(D자재만)",
         실측="전·후 모두 10월 실측 불가 → 가정값에서 나온 결과. 같은 구조의 A3·C2·C3 양중 불출 체류(중앙값 1.9시간)는 참고용으로만 제시(D자재에 직접 적용 불가)"),
    dict(KPI="KPI2", 지표="적치 점유율(%) · 면적 초과일(일)",
         데이터파일="위 적치 체류와 동일 + 15_site_constraints.csv(C08: B3 적치용량 초과 시 하역장 점유 증가)",
         연결키="위와 동일",
         산식=f"점유율 = 그날 적치 물량(대) × LOAD_AREA({A.LOAD_AREA}㎡) ÷ 구역 면적(전 B3 {A.B3_AREA}㎡, 후 B4 {A.B4_AREA}㎡) | 면적 초과일 = 점유율 > 100%인 날 수",
         가정="A10 면적(자료에 없음), A09 체류. 두 값은 'B3가 10월 반입에서 거의 넘치지 않는다'는 검사(C5)를 통과하도록 짝지어 정함",
         제외기준="D자재 외 적치 제외 | 자료에 면적 값이 없어 절대 면적 해석은 불가, 전·후 상대 비교와 C08 방향성에만 의미",
         실측="10월 실측 불가(면적·적치량 기록 없음). 04번 B3 카메라는 사람·지게차 수만 있음"),
]

def header(kpi):
    """보고서 맨 위에 붙이는 명세 요약(해당 KPI만)."""
    out = [f"{'=' * 100}", f"[{kpi} 데이터·산식·가정·제외 명세]  (전체 표: KPI_SPEC.md, 가정 목록: assumptions.py, 모순 검사: check_assumptions.py)", "=" * 100]
    for s in SPEC:
        if s["KPI"] != kpi: continue
        out.append(f"■ {s['지표']}")
        for k in ("데이터파일", "연결키", "산식", "가정", "제외기준", "실측"):
            out.append(f"  - {k}: {s[k]}")
    out.append("가정값 요약: " + "; ".join(f"{r[0]} {r[1]}={r[2]}{r[3]}" for r in A.ASSUMPTIONS if kpi in r[7] and r[4] != "데이터 값") + " ...")
    out.append("=" * 100 + "\n")
    return "\n".join(out)

def build_md():
    md = ["# KPI 명세 — 데이터파일 · 연결키 · 산식 · 가정 · 제외기준 (v2)", "",
          "모든 값은 `assumptions.py`(단일 출처)에서 읽는다. 가정끼리·자료와 모순이 없는지는 `check_assumptions.py`가 자료를 다시 읽어 검사한다(아래 표).", ""]
    for s in SPEC:
        md += [f"## {s['KPI']} — {s['지표']}", ""]
        md += [f"| 항목 | 내용 |", "|---|---|"] + [f"| {k} | {s[k]} |" for k in ("데이터파일", "연결키", "산식", "가정", "제외기준", "실측")] + [""]
    md += ["## 가정 목록 (assumptions.py)", "", "분류: 데이터 값(자료에서 읽음) / 데이터 추정(통계 추정, 구간 있음) / 정책 설계값(해결방안 문서) / 가정(자료 근거 없음)", "", A.table(), ""]
    if os.path.exists("out/check_assumptions.csv"):
        c = pd.read_csv("out/check_assumptions.csv")
        md += ["## 가정 모순 검사 결과 (check_assumptions.py)", "", "| ID | 검사 | 결과 | 근거 |", "|---|---|---|---|"] + [f"| {r.ID} | {r.검사} | {r.결과} | {r.근거} |" for r in c.itertuples()] + [""]
    return "\n".join(md)

if __name__ == "__main__":
    open("KPI_SPEC.md", "w", encoding="utf-8").write(build_md())
    pd.DataFrame(SPEC).to_csv("out/kpi_spec.csv", index=False)
    pd.DataFrame(A.ASSUMPTIONS, columns=["ID", "이름", "값", "단위", "분류", "출처", "근거·범위", "영향KPI"]).to_csv("out/assumptions_v2.csv", index=False)
    print("저장: KPI_SPEC.md, out/kpi_spec.csv, out/assumptions_v2.csv")
