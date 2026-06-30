@echo off
cd /d "%~dp0"
echo ====================================
echo   Level-3 + TRSM Profiling
echo ====================================
set METRICS=gpu__time_duration.sum,dram__bytes.sum

echo === Level-3 (matrix-matrix) ===
for %%S in (256 512 1024 2048) do (
    ncu --metrics %METRICS% --csv benchmark_all.exe gemm %%S %%S %%S > ncu_L3_GEMM_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe symm %%S %%S > ncu_L3_SYMM_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe syrk %%S %%S > ncu_L3_SYRK_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe syr2k %%S %%S > ncu_L3_SYR2K_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe trmm %%S %%S > ncu_L3_TRMM_N%%S.csv 2>&1
    echo   L3 S=%%S done
)

echo.
echo === TRSM ===
for %%S in (256 512 1024 2048) do (
    ncu --metrics %METRICS% --csv benchmark_blas.exe trsm %%S %%S > ncu_TRSM_N%%S.csv 2>&1
    echo   TRSM N=%%S done
)

echo.
echo === Done! ===
dir /b ncu_L3_*.csv ncu_TRSM_*.csv 2>nul
pause
