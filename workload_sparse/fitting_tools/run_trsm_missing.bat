@echo off
cd /d "%~dp0"
echo TRSM missing sizes...
set METRICS=gpu__time_duration.sum,dram__bytes.sum
for %%S in (128 3072 4096) do (
    ncu --metrics %METRICS% --csv benchmark_blas.exe trsm %%S %%S > ncu_TRSM_N%%S.csv 2>&1
    echo   TRSM S=%%S done
)
pause
