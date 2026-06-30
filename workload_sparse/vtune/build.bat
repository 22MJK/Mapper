@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cl /O2 /arch:AVX2 benchmark_trsv_cpu.cpp /Fe:benchmark_trsv_cpu.exe
