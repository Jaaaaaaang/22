"""원본 코드 출력의 이상값 원인 확인 (검사용). 실행: python3 checks2.py"""
import pandas as pd, numpy as np, warnings
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor as GBR
warnings.filterwarnings("ignore")
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",200)
D="data/"
raw=pd.read_csv(D+"05_delivery_log.csv")
T=["Reserved_Start","Reserved_End","Arrival_Time","Gate_In_Time","Unload_Start_Time","Unload_End_Time","Gate_Out_Time"]
for c in T: raw[c]=pd.to_datetime(raw[c],errors="coerce")
day=pd.to_datetime(raw.Date)
for n,c in zip(["rsv","rsve","arr","gin","us","ue","go"],T): raw[n]=(raw[c]-day).dt.total_seconds()/60
g=raw[raw.Truck_ID.str.startswith("TRK")].copy(); rem=raw[~raw.Truck_ID.str.startswith("TRK")].copy()
g["early"]=g.rsv-g.arr; g["ext"]=g.gin-g.arr; g["iw"]=g.us-g.gin; g["unl"]=g.ue-g.us; g["exg"]=g.go-g.ue
g["tz"]=g.Final_Target_Zone; g["eq"]=g.Unloading_Equipment_ID; g["fl"]=g.Unloading_Equipment_ID

print("== [1] ext(도착→입문)는 무엇으로 설명되는가")
g["n_in"]=[int(((g.Date==x.Date)&(g.gin<x.arr)&(g.go>x.arr)).sum()) for x in g.itertuples()]
g["n_wait"]=[int(((g.Date==x.Date)&(g.arr<x.arr)&(g.gin>x.arr)).sum()) for x in g.itertuples()]
g["n_near"]=[int(((g.Date==x.Date)&((g.arr-x.arr).abs()<=15)).sum())-1 for x in g.itertuples()]
g["rem_ov"]=[int(((rem.Date==x.Date)&(rem.gin>=x.arr-5)&(rem.gin<=x.gin+5)).sum()) for x in g.itertuples()]
g["remday"]=g.Date.isin(rem.Date.unique()).astype(int)
print("corr(ext, ·):", g[["ext","early","n_in","n_wait","n_near","rem_ov","remday","rsv","iw","unl"]].corr().ext.round(2).to_dict())
for k in ["n_in","n_wait","n_near","remday"]: print(k, g.groupby(k).ext.agg(["count","mean"]).round(1).to_dict("index"))
print("early>0 vs <=0 의 ext 평균:", g.groupby(g.early>0).ext.mean().round(1).to_dict())
print("ext 분포(분):", g.ext.astype(int).value_counts().sort_index().to_dict())
print("입문−예약시작(분) 분위수:", (g.gin-g.rsv).quantile([.1,.25,.5,.75,.9]).round(0).to_dict())

print("\n== [2] 4대 동시 체류일 시간표")
for dt in ["2026-10-14","2026-10-27"]:
    x=g[g.Date==dt].sort_values("gin"); print(dt); print(x[["Delivery_ID","tz","eq","rsv","arr","gin","us","ue","go","iw","unl"]].astype({c:int for c in ["rsv","arr","gin","us","ue","go","iw","unl"]}).to_string(index=False))

print("\n== [3] D구역 자재를 FL-02가 아닌 지게차가 맡은 경우, 그때 FL-02는 사용 중이었나")
for x in g[(g.tz=="D")&(g["eq"]!="FL-02")].itertuples():
    o=g[(g.Date==x.Date)&(g["eq"]=="FL-02")&(g.us<x.ue)&(g.ue>x.us)]
    print(x.Delivery_ID,x.Date,x[5] if False else getattr(x,"fl","?"),"FL-02 겹침:",len(o), "| 같은 시간 구내:",int(((g.Date==x.Date)&(g.gin<x.us)&(g.go>x.us)).sum())-1)
print("같은 지게차가 연속 배정될 때 앞 하역종료→다음 하역시작 간격(분) 최소:", min((b.us-a.ue) for dt,y in g.dropna(subset=['ue']).groupby(["Date","eq"]) for a,b in zip(y.sort_values("us").itertuples(),y.sort_values("us").iloc[1:].itertuples())))
both=[(dt,len(y)) for dt,y in g.groupby("Date") if (y.tz=="D").sum()>=2]
ovD=0
for dt,y in g.groupby("Date"):
    yd=y[y.tz=="D"].sort_values("gin")
    for a,b in zip(yd.itertuples(),yd.iloc[1:].itertuples()):
        if b.gin<a.go: ovD+=1; print("  D 트럭 2대 체류 겹침:",dt,a.Delivery_ID,a.fl,b.Delivery_ID,b.fl,"뒤 트럭 iw",int(b.iw))
print("D 트럭 체류 겹침 건수:",ovD)

print("\n== [4] iw(입문→하역시작)의 구조")
ok=g[g.iw<=12]; print("iw<=12 인 78건: 평균",round(ok.iw.mean(),2),"| 구역별",ok.groupby("tz").iw.mean().round(1).to_dict(),"| 지게차별",ok.groupby("eq").iw.mean().round(1).to_dict())
print("corr(iw<=12, n_in/unl/rsv):", ok[["iw","n_in","unl","rsv","ext"]].corr().iw.round(2).to_dict())

print("\n== [5] ③ 위험 점수(포아송) 계수와 상한 산출 과정")
c=pd.read_csv(D+"04_camera_object_log.csv"); b2=c[c.Subzone=="B2"].copy(); b2["h"]=pd.to_datetime(b2.Slot_Start).dt.hour
RC=["Truck_Count","Forklift_Count","People_Count"]; R=PoissonRegressor(alpha=1e-3,max_iter=1000).fit(b2[RC],b2.Proximity_Alert_Count)
print("계수(트럭,지게차,사람):",R.coef_.round(3),"절편",round(R.intercept_,3))
PPL=b2.groupby("h").People_Count.mean(); THR=R.predict(b2[RC])[b2.Truck_Count.values>=1].mean()
lam=lambda n,h: float(R.predict(pd.DataFrame([[n,min(n,3),PPL[h]]],columns=RC))[0])
print("원본 기준값 THR(트럭>=1 슬롯 평균 예측):",round(THR,3))
print("lam(k, 10시):",{k:round(lam(k,10),2) for k in (0,1,2,3,4)}, "→ k=2도 THR 초과:", lam(2,10)>THR, "(상한 2는 '조건 만족'이 아니라 기본값)")
ref=float(R.predict(pd.DataFrame([[2,2,5]],columns=RC))[0]); print("해결방안 문서 기준(10/19 09시: 트럭2·지게차2·사람5)의 점수:",round(ref,3))
print("문서 기준 이하가 되는 최대 k(시간대별):",{h:max([k for k in (1,2,3,4) if lam(k,h)<=ref],default=0) for h in range(9,16)})
on=b2[b2.Truck_Count>=1]; R2=PoissonRegressor(alpha=1e-3,max_iter=1000).fit(on[RC],on.Proximity_Alert_Count)
print("트럭>=1 슬롯(90개)만으로 학습한 계수:",R2.coef_.round(3),"| 트럭 수별 실제 경보 평균:",on.groupby("Truck_Count").Proximity_Alert_Count.mean().round(2).to_dict())

print("\n== [6] ① 분위수 회귀 검증(기준선 비교)")
g=g.reset_index(drop=True); dd=pd.to_datetime(g.Date)
def feats(df):
    ns=[int(((df.Date==r.Date)&((df.rsv-r.rsv).abs()<=15)).sum())-1 for r in df.itertuples()]
    return pd.DataFrame({"h":df.rsv.values/60,"D":(df.tz=="D").astype(int).values,"dow":pd.to_datetime(df.Date).dt.dayofweek.values,"ns":ns})
tr,te=g[g.Date<="2026-10-21"],g[g.Date>"2026-10-21"]
P={q:GBR(loss="quantile",alpha=q,n_estimators=80,max_depth=2,min_samples_leaf=3,random_state=0).fit(feats(tr),tr.early).predict(feats(te)) for q in (.1,.5,.9)}
pin=lambda y,p,q: float(np.mean(np.maximum(q*(y-p),(q-1)*(y-p))))
y=te.early.values
print("검증 36건 MAE: 모형 중앙값",round(np.abs(y-P[.5]).mean(),1),"| 학습기간 중앙값 상수",round(np.abs(y-tr.early.median()).mean(),1),"| 학습기간 평균 상수",round(np.abs(y-tr.early.mean()).mean(),1))
for q in (.1,.5,.9): print(f" 핀볼손실 q={q}: 모형 {pin(y,P[q],q):.2f} vs 학습기간 분위수 상수 {pin(y,tr.early.quantile(q),q):.2f}")
print(" 80% 구간 포함률: 모형",round(np.mean((y>=np.minimum(P[.1],P[.9]))&(y<=np.maximum(P[.1],P[.9]))),2),"| 상수 분위수",round(np.mean((y>=tr.early.quantile(.1))&(y<=tr.early.quantile(.9))),2))
print(" 예측 폭(q90-q10) 분포:",np.round(np.percentile(np.abs(P[.9]-P[.1]),[10,50,90]),1),"| 학습기간 early 분위수 10/50/90:",tr.early.quantile([.1,.5,.9]).round(0).tolist(),"| 검증기간:",te.early.quantile([.1,.5,.9]).round(0).tolist())

print("\n== [7] ④ 풍속→보류 로지스틱")
W=pd.read_csv(D+"13_weather_hourly.csv"); mx=W.groupby("Date").Wind_mps.max(); L=pd.read_csv(D+"06_lifting_log.csv")
for nm,hd in [("원본('보류' 상태, 10/22 포함)",set(L[L.Status.str.contains("보류")].Date)),("수정(사유=풍속)",set(L[L.Hold_Reason.str.contains("풍속")].Date))]:
    yy=[int(x in hd) for x in mx.index]; M=LogisticRegression(C=10).fit(mx.values.reshape(-1,1),yy)
    print(nm,"보류일",sum(yy),"| P(10.0)=",round(M.predict_proba([[10.0]])[0,1],2),"| P(12.4/12.7/13.0)=",M.predict_proba([[12.4],[12.7],[13.0]])[:,1].round(2),"| P(7.3)=",round(M.predict_proba([[7.3]])[0,1],2))
print("지상 10.0 m/s일 때 지브 보정풍속: TC-01",10*1.35,"TC-02",10*1.25,"MC-01",10*1.15,"→ 기준(12/12/10) 모두 초과")
