@echo off
cd /d "%~dp0"
echo Profiling TRSM only...
set METRICS=gpu__time_duration.sum,dram__bytes.sum
for %%S in (256 512 1024 2048) do (
    ncu --metrics %METRICS% --csv benchmark_blas.exe trsm %%S %%S > ncu_TRSM_N%%S.csv 2>&1
    echo   TRSM N=%%S done
)
echo Done!
pause
