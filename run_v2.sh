#!/bin/bash
# v2 가정(assumptions.py)으로 전체 재실행. CPU 2코어 기준 체인 2개 병렬.
#  체인 A: bflow_sim.py 본 실행 100 + 민감도 4종(각 30, 앞 30개 시나리오와 같은 시드)
#  체인 B: kpi2_forecast.py 본 실행 100 + 민감도 4종(체류 v1 값, 체류 길게, 반입 탄력성 하한/상한)
cd /home/user/webapp
( N_RUN=100 TAG=main python3 bflow_sim.py > out/main_n100.txt 2>&1
  WINDOW=own   N_RUN=30 VAL_REP=5 TAG=sens_window_own python3 bflow_sim.py > out/sens_window_own.txt 2>&1
  CAP=3        N_RUN=30 VAL_REP=5 TAG=sens_cap3       python3 bflow_sim.py > out/sens_cap3.txt 2>&1
  QMODE=const  N_RUN=30 VAL_REP=5 TAG=sens_qconst     python3 bflow_sim.py > out/sens_qconst.txt 2>&1
  P_COMPLY=0.5 N_RUN=30 VAL_REP=5 TAG=sens_p05        python3 bflow_sim.py > out/sens_p05.txt 2>&1
  date > out/CHAIN_A_DONE ) &
( N_RUN=100 TAG=main python3 kpi2_forecast.py > out/kpi2_run_main.txt 2>&1
  N_RUN=30 TAG=sens_stay_v1   STAY_SET=1,2,3 python3 kpi2_forecast.py > out/kpi2_run_sens_stay_v1.txt 2>&1
  N_RUN=30 TAG=sens_stay_long STAY_SET=2,3,4 python3 kpi2_forecast.py > out/kpi2_run_sens_stay_long.txt 2>&1
  N_RUN=30 TAG=sens_beta_lo   BETA_TRK=0.33  python3 kpi2_forecast.py > out/kpi2_run_sens_beta_lo.txt 2>&1
  N_RUN=30 TAG=sens_beta_hi   BETA_TRK=1.27  python3 kpi2_forecast.py > out/kpi2_run_sens_beta_hi.txt 2>&1
  date > out/CHAIN_B_DONE ) &
wait
