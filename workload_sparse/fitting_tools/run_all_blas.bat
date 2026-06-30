@echo off
cd /d "%~dp0"
echo ====================================
echo   BLAS L2+L3 Profiling -- Admin Required!
echo ====================================
echo.
echo If you see ERR_NVGPUCTRPERM below:
echo   RIGHT-CLICK this file -> Run as Administrator
echo.
pause
echo.

set METRICS=gpu__time_duration.sum,dram__bytes.sum

echo === Level-2 (matrix-vector) ===
for %%N in (256 512 1024 2048 4096) do (
    ncu --metrics %METRICS% --csv benchmark_all.exe gemv %%N %%N > ncu_L2_GEMV_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe symv %%N > ncu_L2_SYMV_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe trmv %%N > ncu_L2_TRMV_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe ger %%N %%N > ncu_L2_GER_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe syr %%N > ncu_L2_SYR_N%%N.csv 2>&1
    echo   L2 N=%%N done
)

echo.
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
echo ====================================
echo   Done!
dir /b ncu_L*.csv 2>nul
echo ====================================
pause
