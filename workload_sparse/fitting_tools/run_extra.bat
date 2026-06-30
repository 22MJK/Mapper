@echo off
cd /d "%~dp0"
echo ====================================
echo   Extra Operators Profiling (Admin!)
echo ====================================
pause
set METRICS=gpu__time_duration.sum,dram__bytes.sum

echo === gbmv (KL=KU=N/4 auto) ===
for %%N in (256 512 1024 2048 4096 6144 8192) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe gbmv %%N > ncu_GBMV_N%%N.csv 2>&1
    echo   gbmv N=%%N done
)

echo === Sparse ops (nnz/row=8) ===
for %%N in (1024 2048 4096 8192 16384 32768 65536) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe spmv %%N 8 > ncu_SPMV_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_extra.exe sptrsv %%N 8 > ncu_SPTRSV_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_extra.exe t_sparse %%N 8 > ncu_TSPARSE_N%%N.csv 2>&1
    echo   sparse N=%%N done
)

echo === spgemm (nnz/row=4) ===
for %%N in (512 1024 2048 4096 6144 8192 10240) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe spgemm %%N 4 > ncu_SPGEMM_N%%N.csv 2>&1
    echo   spgemm N=%%N done
)

echo === dense transpose ===
for %%N in (128 256 512 1024 2048 4096 6144) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe t_dense %%N %%N > ncu_TDENSE_N%%N.csv 2>&1
    echo   t_dense N=%%N done
)

echo === LAPACK (potrf/getrf/geqrf) ===
for %%N in (128 256 512 1024 2048 3072 4096) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe potrf %%N > ncu_POTRF_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_extra.exe getrf %%N > ncu_GETRF_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_extra.exe geqrf %%N %%N > ncu_GEQRF_N%%N.csv 2>&1
    echo   lapack N=%%N done
)

echo === Done! ===
pause
