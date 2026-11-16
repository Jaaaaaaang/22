"""코드 가정 vs 데이터 대조 (검사용). 실행: python3 checks.py"""
import pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",200)
D="data/"
raw=pd.read_csv(D+"05_delivery_log.csv")
T=["Reserved_Start","Reserved_End","Arrival_Time","Gate_In_Time","Unload_Start_Time","Unload_End_Time","Gate_Out_Time"]
for c in T: raw[c]=pd.to_datetime(raw[c],errors="coerce")
day=pd.to_datetime(raw.Date)
for n,c in zip(["rsv","rsve","arr","gin","us","ue","go"],T): raw[n]=(raw[c]-day).dt.total_seconds()/60
g=raw[raw.Truck_ID.str.startswith("TRK")].copy(); r=raw[~raw.Truck_ID.str.startswith("TRK")].copy()
g["early"]=g.rsv-g.arr; g["early_gin"]=g.rsv-g.gin; g["ext"]=g.gin-g.arr; g["iw"]=g.us-g.gin; g["unl"]=g.ue-g.us; g["exg"]=g.go-g.ue
print("== [A] 05 일반트럭 기본")
print(g[["early","early_gin","ext","iw","unl","exg"]].describe().round(1).T)
print("결측 행:", g[g.ue.isna()][["Delivery_ID","Date","Unload_Start_Time","Unload_End_Time","Gate_Out_Time"]].to_string(index=False))
print("문서번호 중복:", raw[raw.Source_Document_No.duplicated(keep=False)][["Delivery_ID","Source_Document_No","Truck_ID","Date"]].to_string(index=False))
print("조기도착(early>0):",(g.early>0).sum(),"/",len(g), "| 예약 대비 입문 조기(early_gin>0):",(g.early_gin>0).sum())
print("early>60분:", g[g.early>60][["Delivery_ID","Date","early","ext"]].round(0).to_string(index=False))

print("\n== [B] C01 예약창(09-11,14-16 시작) 준수")
inwin=lambda m: ((m>=540)&(m<=660))|((m>=840)&(m<=960))
print("예약 시작이 C01 창 밖:", (~inwin(g.rsv)).sum(), "/", len(g), "| 입문이 창 밖:", (~inwin(g.gin)).sum())
print("예약 시작 시(hour) 분포:", (g.rsv//60).astype(int).value_counts().sort_index().to_dict())
print("입문 시(hour) 분포:", (g.gin//60).astype(int).value_counts().sort_index().to_dict())
print("예약창 길이(분):", (g.rsve-g.rsv).value_counts().to_dict())
print("입문이 자기 예약창 [rsv,rsve] 안:", ((g.gin>=g.rsv)&(g.gin<=g.rsve)).sum())

print("\n== [C] 하역 동시성(하역자리·지게차)")
def maxov(s,e):
    ev=sorted([(a,1) for a in s]+[(b,-1) for b in e]); n=m=0
    for _,k in ev: n+=k; m=max(m,n)
    return m
gg=g.dropna(subset=["ue"])
mo=gg.groupby("Date").apply(lambda x: maxov(x.us,x.ue)); print("일별 최대 동시 하역 대수 분포:", mo.value_counts().sort_index().to_dict())
fo=gg.groupby(["Date","Unloading_Equipment_ID"]).apply(lambda x: maxov(x.us,x.ue)); print("지게차 1대의 동시 하역 최대:", fo.value_counts().sort_index().to_dict())
ms=g.groupby("Date").apply(lambda x: maxov(x.gin,x.go)); print("일별 최대 동시 체류(입문~출문):", ms.value_counts().sort_index().to_dict(), "| 4대 날:", list(ms[ms>=4].index), "| 3대 이상 날:", list(ms[ms>=3].index))
print("지게차×목적지:"); print(pd.crosstab(g.Unloading_Equipment_ID,g.Final_Target_Zone))
print("iw 분포:", g.iw.round(0).value_counts().sort_index().to_dict())
# 입문 시점 구내 대수별 iw
g["n_in"]=[int(((g.Date==x.Date)&(g.gin<x.gin)&(g.go>x.gin)).sum()) for x in g.itertuples()]
g["fl_busy"]=[int(((g.Date==x.Date)&(g.Unloading_Equipment_ID==x.Unloading_Equipment_ID)&(g.us<x.gin)&(g.ue>x.gin)).sum()) for x in g.itertuples()]
print("입문 시 구내 트럭 수별 iw 평균:"); print(g.groupby("n_in").iw.agg(["count","mean","min","max"]).round(1))
print("입문 시 자기 지게차 사용중 여부별 iw:"); print(g.groupby("fl_busy").iw.agg(["count","mean","min","max"]).round(1))
print("iw>20:", g[g.iw>20][["Delivery_ID","Date","iw","n_in","fl_busy","Unloading_Equipment_ID"]].to_string(index=False))

print("\n== [D] 일자별 트럭 수(구역별) — 수요 추정 구간(10/14~) 확인")
ct=g.groupby(["Date","Final_Target_Zone"]).size().unstack(fill_value=0); ct["합"]=ct.sum(axis=1); print(ct.T.to_string())
w=pd.read_csv(D+"02_daily_work_plan.csv"); alld=sorted(w.Date.unique())
print("트럭 없는 작업일:", [x for x in alld if x not in ct.index])
print("레미콘 타설일 입문 시각:", {k:[int(v) for v in x.gin] for k,x in r.groupby("Date")})
print("02 건물별 계획인원 일평균(10/19~31):"); print(w[w.Date>="2026-10-19"].groupby(["Date","Building"]).Planned_Workers.sum().unstack().mean().round(1).to_dict())

print("\n== [E] 04 B2 카메라: 트럭 수별 근접경보")
c=pd.read_csv(D+"04_camera_object_log.csv"); b2=c[c.Subzone=="B2"].copy(); b2["h"]=pd.to_datetime(b2.Slot_Start).dt.hour
print(b2.groupby("Truck_Count").Proximity_Alert_Count.agg(["count","mean","sum"]).round(2))
print("지게차 수별:"); print(b2.groupby("Forklift_Count").Proximity_Alert_Count.agg(["count","mean"]).round(2))
print("트럭>=1 슬롯에서 corr(alert, truck/forklift/people):", b2[b2.Truck_Count>=1][["Truck_Count","Forklift_Count","People_Count","Proximity_Alert_Count"]].corr().Proximity_Alert_Count.round(2).to_dict())
# 카메라 트럭 수 vs 05 입문~출문 동시대수
def conc(dt,h):
    x=g[g.Date==dt]; return max((int(((x.gin<=t)&(x.go>t)).sum()) for t in range(h*60,h*60+60)), default=0)
b2["c05"]=[conc(a,b) for a,b in zip(b2.Date,b2.h)]
print("카메라 Truck_Count == 05 입문~출문 시간내 최대 동시대수 일치율:", round((b2.Truck_Count==b2.c05).mean(),3)); print(pd.crosstab(b2.Truck_Count,b2.c05))
print("10/19 09시 B2:", b2[(b2.Date=="2026-10-19")&(b2.h==9)][["People_Count","Forklift_Count","Truck_Count","Proximity_Alert_Count"]].to_dict("records"))

print("\n== [F] 06 보류일 vs 13 일최대풍속")
L=pd.read_csv(D+"06_lifting_log.csv"); W=pd.read_csv(D+"13_weather_hourly.csv"); mx=W.groupby("Date").Wind_mps.max()
a=set(L[L.Status.str.contains("보류")].Date); b=set(L[L.Hold_Reason.str.contains("풍속")].Date)
print("'보류' 포함 날(원본 코드 기준):", sorted(a)); print("사유가 풍속인 날:", sorted(b)); print("풍속 아닌 보류:", L[(L.Status!="완료")&~L.Hold_Reason.str.contains("풍속")][["Lift_ID","Date","Status","Hold_Reason"]].to_dict("records"))
print("일최대풍속 >=10 날:", mx[mx>=10].to_dict(), "| 그 외 최대:", mx[mx<10].max())
print("지상풍속 환산 중지기준: TC-01", round(12/1.35,2), "TC-02", round(12/1.25,2), "MC-01", round(10/1.15,2))
hrs=W[W.Wind_mps>=8.7].groupby("Date").size(); print("지상 8.7m/s 이상 시간 수(일별):", hrs.to_dict())
