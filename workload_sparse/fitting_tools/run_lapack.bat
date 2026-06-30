@echo off
cd /d "%~dp0"
echo === LAPACK (potrf/getrf/geqrf) ===
set METRICS=gpu__time_duration.sum,dram__bytes.sum

for %%N in (128 256 512 1024 2048 3072 4096) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe potrf %%N > ncu_POTRF_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_extra.exe getrf %%N > ncu_GETRF_N%%N.csv 2>&1
    ncu --metrics %METRICS% --csv benchmark_extra.exe geqrf %%N %%N > ncu_GEQRF_N%%N.csv 2>&1
    echo   LAPACK N=%%N done
)

echo Done!
pause
