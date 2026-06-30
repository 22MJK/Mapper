@echo off
cd /d "%~dp0"
echo Profiling missing L3+TRSM sizes...
set METRICS=gpu__time_duration.sum,dram__bytes.sum

echo === L3 missing (128, 3072, 4096) ===
for %%S in (128 3072 4096) do (
    ncu --metrics %METRICS% --csv benchmark_all.exe gemm %%S %%S %%S > ncu_L3_GEMM_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe symm %%S %%S > ncu_L3_SYMM_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe syrk %%S %%S > ncu_L3_SYRK_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe syr2k %%S %%S > ncu_L3_SYR2K_N%%S.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_all.exe trmm %%S %%S > ncu_L3_TRMM_N%%S.csv 2>&1
    echo   L3 S=%%S done
)

echo === TRSM missing (128, 3072, 4096) ===
for %%S in (128 3072 4096) do (
    ncu --metrics %METRICS% --csv benchmark_blas.exe trsm %%S %%S > ncu_TRSM_N%%S.csv 2>&1
    echo   TRSM S=%%S done
)

echo Done!
pause
