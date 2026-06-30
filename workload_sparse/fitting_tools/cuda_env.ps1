# cuda_env.ps1 — 在当前终端初始化 CUDA 12.6 + MSVC 编译环境
# 用法: . .\cuda_env.ps1
# 或从 VS Code 终端: & ".\cuda_env.ps1"

$VsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$VsDevShell = Join-Path $VsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"

if (Test-Path $VsDevShell) {
    Import-Module $VsDevShell -Force
    Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
}

# 确认 CUDA
if (Test-Path $env:CUDA_PATH) {
    $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
}

Write-Host "=== CUDA 环境已就绪 ===" -ForegroundColor Green
Write-Host "CUDA:  $(nvcc --version 2>$null | Select-Object -First 3 | Out-String)" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null | ForEach-Object { Write-Host "GPU:   $_" -ForegroundColor Cyan }
Write-Host "MSVC:  $(cl.exe 2>$null | Select-Object -First 1)" -ForegroundColor Cyan
Write-Host ""
Write-Host "用法: nvcc hello.cu -o hello.exe" -ForegroundColor Yellow
