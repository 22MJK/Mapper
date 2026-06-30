@echo off
cd /d "%~dp0"
echo ====================================
echo   SYRK + TRSM Profiling
echo ====================================
echo.

set METRICS=gpu__time_duration.sum,dram__bytes.sum

echo --- SYRK (C = A*A^T, N=K) ---
for %%N in (128 256 512 1024 2048) do (
    echo [*] SYRK N=K=%%N ...
    ncu --metrics %METRICS% --csv benchmark_blas.exe syrk %%N %%N > ncu_SYRK_N%%N.csv 2>&1
    echo     Done
)

echo.
echo --- TRSM (solve T*X=B, N=M) ---
for %%N in (128 256 512 1024 2048) do (
    echo [*] TRSM N=M=%%N ...
    ncu --metrics %METRICS% --csv benchmark_blas.exe trsm %%N %%N > ncu_TRSM_N%%N.csv 2>&1
    echo     Done
)

echo.
echo ====================================
echo   Done!
dir /b ncu_SYRK_*.csv ncu_TRSM_*.csv 2>nul
echo ====================================
pause
