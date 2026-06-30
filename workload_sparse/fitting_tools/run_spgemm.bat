@echo off
cd /d "%~dp0"
echo === SPGEMM only ===
set METRICS=gpu__time_duration.sum,dram__bytes.sum
for %%N in (512 1024 2048 4096 6144 8192 10240) do (
    ncu --metrics %METRICS% --csv benchmark_extra.exe spgemm %%N 4 > ncu_SPGEMM_N%%N.csv 2>&1
    echo   spgemm N=%%N done
)
pause
