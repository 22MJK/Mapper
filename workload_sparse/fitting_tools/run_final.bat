@echo off
cd /d "%~dp0"
echo === Final missing: SPTRSV + TDENSE + SPGEMM ===
set METRICS=gpu__time_duration.sum,dram__bytes.sum

for %%N in (1024 2048 4096 8192 16384 32768 65536) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe sptrsv %%N 8 > ncu_SPTRSV_N%%N.csv 2>&1
    echo   sptrsv N=%%N done
)

for %%N in (512 1024 2048 4096 6144 8192 10240) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe spgemm %%N 4 > ncu_SPGEMM_N%%N.csv 2>&1
    echo   spgemm N=%%N done
)

for %%N in (128 256 512 1024 2048 4096 6144) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe t_dense %%N %%N > ncu_TDENSE_N%%N.csv 2>&1
    echo   t_dense N=%%N done
)

echo Done!
pause
