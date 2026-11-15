"""수정 전 확인 3건 (검사용). 실행: python3 checks3.py"""
import pandas as pd, numpy as np, warnings
from sklearn.ensemble import GradientBoostingRegressor as GBR
warnings.filterwarnings("ignore")
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
D="data/"
raw=pd.read_csv(D+"05_delivery_log.csv")
T=["Reserved_Start","Reserved_End","Arrival_Time","Gate_In_Time","Unload_Start_Time","Unload_End_Time","Gate_Out_Time"]
for c in T: raw[c]=pd.to_datetime(raw[c],errors="coerce")
day=pd.to_datetime(raw.Date)
for n,c in zip(["rsv","rsve","arr","gin","us","ue","go"],T): raw[n]=(raw[c]-day).dt.total_seconds()/60
g=raw[raw.Truck_ID.str.startswith("TRK")].copy().reset_index(drop=True)
g["early"]=g.rsv-g.arr; g["ext"]=g.gin-g.arr; g["iw"]=g.us-g.gin; g["unl"]=g.ue-g.us
g["tz"]=g.Final_Target_Zone; g["eq"]=g.Unloading_Equipment_ID; g["fl"]=g.Unloading_Equipment_ID

print("== [1] '.eq' 속성 문제")
row=g.iloc[0]; print("row['eq'] =", row["eq"], "| row.eq 의 타입 =", type(row.eq).__name__, "| row.eq in {'FL-01','FL-02','FL-03'} →", (row.eq in {"FL-01":0,"FL-02":0,"FL-03":0}))
print("→ 원본 execute()의 `r = df.iloc[j]; r.eq` 는 지게차 ID가 아니라 함수 → 항상 'FL-01'로 대체됨")
def replay_iw(assign):
    out=[]
    for dt,x in g.groupby("Date"):
        free={}
        for r in x.sort_values("gin").itertuples():
            f=assign(r); us=max(r.gin, free.get(f,0.0)); ue=us+(r.unl if r.unl==r.unl else 38.3); free[f]=ue; out.append(us-r.gin)
    return np.array(out)
a=replay_iw(lambda r:"FL-01"); b=replay_iw(lambda r:r.fl)
print("지게차 대기만 본 10월 재현(분): 전부 FL-01 → 평균",a.round(1).mean().round(1),"최대",a.max(),"대기>0",int((a>0).sum()),"건 | 실제 배정 → 평균",b.mean().round(2),"최대",b.max(),"대기>0",int((b>0).sum()),"건")

print("\n== [2] D자재를 다른 지게차가 맡았을 때 FL-02 상태 (컬럼명 수정 후 재확인)")
for x in g[(g.tz=="D")&(g.fl!="FL-02")].itertuples():
    o=g[(g.Date==x.Date)&(g.fl=="FL-02")&(g.us<x.us+1)&(g.ue>x.gin)]
    print(" ",x.Delivery_ID,x.Date,x.fl,"| 입문~하역시작 사이 FL-02 사용 중:",len(o)>0)
print("지게차 대기(자기 지게차가 사용 중이라 기다림) 2건: 10/14 DL-0020(FL-01 651 종료→652 시작), 10/27 DL-0062(FL-03 925 종료→926 시작)")

print("\n== [3] 게이트 수속 시간: 입문 − max(도착, 예약시작)")
g["proc"]=np.where(g.gin>=g.rsv, g.gin-np.maximum(g.arr,g.rsv), g.gin-g.arr)
g["grp"]=np.where(g.gin<g.rsv,"예약 전 입문",np.where(g.arr>=g.rsv,"예약 후 도착","조기 도착·예약 후 입문"))
print(g.groupby("grp")[["ext","proc"]].agg(["count","mean","median","min","max"]).round(1).to_string())
print("전체 proc 분위수:",g.proc.quantile([0,.1,.25,.5,.75,.9,1]).round(0).tolist(),"| 평균",round(g.proc.mean(),1),"| ext 평균",round(g.ext.mean(),1))
print("corr(proc, 동시 도착/구내 대수):", round(np.corrcoef(g.proc,[int(((g.Date==x.Date)&(g.gin<x.arr)&(g.go>x.arr)).sum()) for x in g.itertuples()])[0,1],2))

print("\n== [4] ①에 업체·자재를 넣으면 나아지는가 (10/05~21 학습 → 10/22~31 검증)")
def base(df):
    ns=[int(((df.Date==r.Date)&((df.rsv-r.rsv).abs()<=15)).sum())-1 for r in df.itertuples()]
    return pd.DataFrame({"h":df.rsv.values/60,"D":(df.tz=="D").astype(int).values,"dow":pd.to_datetime(df.Date).dt.dayofweek.values,"ns":ns},index=df.index)
X0=base(g); X1=pd.concat([X0,pd.get_dummies(g.Receiving_Team,dtype=int),pd.get_dummies(g.Material_Category,dtype=int)],axis=1)
tr=g.Date<="2026-10-21"; y=g.early; pin=lambda y,p,q: float(np.mean(np.maximum(q*(y-p),(q-1)*(y-p))))
for nm,X in [("현재 입력(시각·D여부·요일·동시예약)",X0),("+업체·자재",X1)]:
    s=[]
    for q in (.1,.5,.9):
        p=GBR(loss="quantile",alpha=q,n_estimators=80,max_depth=2,min_samples_leaf=3,random_state=0).fit(X[tr],y[tr]).predict(X[~tr]); s.append(pin(y[~tr].values,p,q))
    print(f" {nm}: 핀볼손실 q10/50/90 = {s[0]:.2f}/{s[1]:.2f}/{s[2]:.2f}")
print(" 상수(학습기간 분위수): 핀볼손실 q10/50/90 =","/".join(f"{pin(y[~tr].values,y[tr].quantile(q),q):.2f}" for q in (.1,.5,.9)))
tr2=tr&(y.abs()<60); print(" 학습기간 조기 60분 초과 3건(10/06) 제외 시 현재 입력:", "/".join(f"{pin(y[~tr].values,GBR(loss='quantile',alpha=q,n_estimators=80,max_depth=2,min_samples_leaf=3,random_state=0).fit(X0[tr2],y[tr2]).predict(X0[~tr]),q):.2f}" for q in (.1,.5,.9)))
print(" 업체별 조기 중앙값(건수):", g.groupby("Receiving_Team").early.agg(["count","median"]).round(0).to_dict("index"))
