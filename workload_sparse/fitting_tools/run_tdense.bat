@echo off
cd /d "%~dp0"
echo === TDENSE only ===
set METRICS=gpu__time_duration.sum,dram__bytes.sum
for %%N in (128 256 512 1024 2048 4096 6144) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe t_dense %%N %%N > ncu_TDENSE_N%%N.csv 2>&1
    echo   t_dense N=%%N done
)
pause
