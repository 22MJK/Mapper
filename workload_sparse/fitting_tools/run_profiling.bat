@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ====================================
echo   TRSV + GEMM Benchmark -- Nsight Compute
echo   GPU: RTX 4060 Laptop (SM 8.9)
echo ====================================
echo.

set METRICS=gpu__time_duration.sum,sm__cycles_elapsed.sum,dram__bytes.sum,lts__t_bytes.sum,sm__throughput.sum.pct_of_peak_sustained_elapsed

echo --- TRSV ---
for %%N in (128 256 512 1024 2048 4096 8192) do (
    echo [*] TRSV N=%%N ...
    ncu --metrics %METRICS% --csv benchmark_trsv.exe %%N > ncu_TRSV_N%%N.csv 2>&1
    echo     Done
)
echo.

echo --- GEMM ---
for %%S in (512 1024 2048) do (
    echo [*] GEMM M=N=K=%%S ...
    ncu --metrics %METRICS% --csv benchmark_gemm.exe %%S %%S %%S > ncu_GEMM_N%%S.csv 2>&1
    echo     Done
)

echo.
echo ====================================
echo   All done!
dir /b ncu_*.csv
echo ====================================
pause
pause
