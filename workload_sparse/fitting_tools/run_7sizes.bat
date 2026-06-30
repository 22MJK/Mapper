@echo off
cd /d "%~dp0"
echo ====================================
echo   All BLAS 7-size Profiling (Admin!)
echo ====================================
echo.
pause
set METRICS=gpu__time_duration.sum,dram__bytes.sum

echo === Level-2 (7 sizes: 128..8192) ===
for %%N in (128 256 512 1024 2048 4096 8192) do (
    ncu --metrics %METRICS% --csv benchmark_all.exe gemv %%N %%N > ncu_L2_GEMV_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe symv %%N > ncu_L2_SYMV_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe trmv %%N > ncu_L2_TRMV_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe ger %%N %%N > ncu_L2_GER_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe syr %%N > ncu_L2_SYR_N%%N.csv 2>&1
    echo   L2 N=%%N done
)

echo.
echo === Level-3 (7 sizes: 128..4096) ===
for %%S in (128 256 512 1024 2048 3072 4096) do (
    ncu --metrics %METRICS% --csv benchmark_all.exe gemm %%S %%S %%S > ncu_L3_GEMM_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe symm %%S %%S > ncu_L3_SYMM_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe syrk %%S %%S > ncu_L3_SYRK_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe syr2k %%S %%S > ncu_L3_SYR2K_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe trmm %%S %%S > ncu_L3_TRMM_N%%S.csv 2>&1
    echo   L3 S=%%S done
)

echo.
echo === TRSM (7 sizes) ===
for %%S in (128 256 512 1024 2048 3072 4096) do (
    ncu --metrics %METRICS% --csv benchmark_blas.exe trsm %%S %%S > ncu_TRSM_N%%S.csv 2>&1
    echo   TRSM S=%%S done
)

echo.
echo === Done! ===
dir /b ncu_L2_*.csv ncu_L3_*.csv ncu_TRSM_*.csv 2>nul | find /c ".csv"
echo files generated.
pause
