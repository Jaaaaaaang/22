#!/bin/bash
# KPI 2 본 실행(100개 시나리오) + 체류일 가정 민감도 2종(각 30개, 본 실행의 앞 30개와 같은 시드). CPU 2코어 기준 병렬 2개.
cd /home/user/webapp
( N_RUN=100 TAG=main python3 kpi2_forecast.py > out/kpi2_run_main.txt 2>&1 ) &
( N_RUN=30 TAG=sens_stay_short STAY_SET=1,1,2 python3 kpi2_forecast.py > out/kpi2_run_sens_stay_short.txt 2>&1
  N_RUN=30 TAG=sens_stay_long  STAY_SET=2,3,4 python3 kpi2_forecast.py > out/kpi2_run_sens_stay_long.txt 2>&1 ) &
wait
date > out/KPI2_ALL_DONE
