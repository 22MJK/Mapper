@echo off
cd /d "%~dp0"
echo === Level-1 Profiling ===
set METRICS=gpu__time_duration.sum,dram__bytes.sum
for %%O in (scal copy axpy dot nrm2 rot rotm) do (
    for %%N in (262144 1048576 4194304 8388608 16777216) do (
        ncu --metrics %METRICS% --csv benchmark_all.exe %%O %%N > ncu_L1_%%O_N%%N.csv 2>&1
    )
    echo   L1 %%O done
)
echo Done!
pause
