@echo off
cd /d "%~dp0"
echo [%time%] Starting GEMM profiling...
echo.

ncu --metrics gpu__time_duration.sum,dram__bytes.sum --csv benchmark_gemm.exe 512 512 512 > ncu_GEMM_test.csv 2>&1

echo [%time%] Exit code: %ERRORLEVEL%
echo.
if exist ncu_GEMM_test.csv (echo FILE OK) else (echo FILE MISSING!)
echo.
pause
