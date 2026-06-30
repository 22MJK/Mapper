@echo off
cd /d "%~dp0"
echo ====================================
echo   GEMM Profiling -- Nsight Compute
echo ====================================
echo.

set METRICS=gpu__time_duration.sum,dram__bytes.sum

for %%S in (128 256 512 1024 2048) do (
    echo [*] GEMM M=N=K=%%S ...
    ncu --metrics %METRICS% --csv benchmark_gemm.exe %%S %%S %%S > ncu_GEMM_N%%S.csv 2>&1
    echo     Done
)

echo.
echo ====================================
echo   Done!
dir /b ncu_GEMM_*.csv 2>nul
echo ====================================
pause
