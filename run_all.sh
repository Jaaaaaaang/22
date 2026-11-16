#!/bin/bash
# 본 실행(100개 시나리오) + 민감도 4종(각 30개 시나리오, 본 실행의 앞 30개와 같은 시드)
cd /home/user/webapp
N_RUN=100 TAG=main python3 bflow_sim.py > out/main_n100.txt 2>&1 &
( WINDOW=own   N_RUN=30 VAL_REP=5 TAG=sens_window_own python3 bflow_sim.py > out/sens_window_own.txt 2>&1
  CAP=3        N_RUN=30 VAL_REP=5 TAG=sens_cap3       python3 bflow_sim.py > out/sens_cap3.txt 2>&1
  QMODE=const  N_RUN=30 VAL_REP=5 TAG=sens_qconst     python3 bflow_sim.py > out/sens_qconst.txt 2>&1
  P_COMPLY=0.5 N_RUN=30 VAL_REP=5 TAG=sens_p05        python3 bflow_sim.py > out/sens_p05.txt 2>&1 ) &
wait
date > out/ALL_DONE
