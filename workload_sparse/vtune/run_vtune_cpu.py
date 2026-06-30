"""
run_vtune_cpu.py — 批量 VTune CPU 剖析 TRSV, 提取 DRAM Bytes
用法 (以管理员身份在 oneAPI 终端运行):
    python run_vtune_cpu.py

前提: benchmark_trsv_cpu.exe 已编译, VTune CLI 可用
"""

import os, sys, subprocess, re, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(SCRIPT_DIR, "benchmark_trsv_cpu.exe")
VTUNE = "vtune"

N_values = [512, 1024, 2048, 4096, 6144, 8192, 10240]

results = []

for N in N_values:
    result_dir = os.path.join(SCRIPT_DIR, f"r_trsv_cpu_N{N}")
    print(f"\n[*] Profiling N={N} ...")

    # 运行 VTune Memory Access 分析
    cmd = [
        VTUNE, "-collect", "uarch-exploration",
        "-result-dir", result_dir,
        "-quiet",
        "--", EXE, str(N), "--no-pause"
    ]
    subprocess.run(cmd, check=True)

    # 提取 DRAM 总量
    report_cmd = [
        VTUNE, "-report", "summary",
        "-result-dir", result_dir,
        "-format", "csv",
        "-report-output", os.path.join(SCRIPT_DIR, f"vtune_summary_N{N}.csv"),
        "-quiet"
    ]
    subprocess.run(report_cmd, check=True)

    # 解析 CSV
    dram_gb = 0.0
    csv_path = os.path.join(SCRIPT_DIR, f"vtune_summary_N{N}.csv")
    with open(csv_path, "r") as f:
        for line in f:
            if "DRAM Data Transferred" in line or "DRAM" in line:
                parts = line.split(",")
                for p in parts:
                    try: dram_gb = float(p); break
                    except: pass

    dram_bytes = dram_gb * 1e9
    flops_theory = N * N  # TRSV ≈ N² FLOPs
    print(f"  N={N}: DRAM={dram_bytes:.0f} B,  FLOPs(理论)={flops_theory:.0f}")

    results.append({"N": N, "DRAM_Bytes": dram_bytes, "FLOPs": flops_theory})

# 输出汇总
print("\n" + "="*60)
print(f"  {'N':>8}  {'DRAM_实测(B)':>16}  {'FLOPs_理论':>14}  {'Bytes/N^2':>10}")
print("-"*60)
for r in results:
    N = r["N"]
    b = r["DRAM_Bytes"]
    f = r["FLOPs"]
    print(f"  {N:>8}  {b:>16.0f}  {f:>14.0f}  {b/(N*N):>10.2f}")

# 拟合 k
Ns = np.array([r["N"] for r in results])
Bs = np.array([r["DRAM_Bytes"] for r in results if r["DRAM_Bytes"] > 0])
Ns_fit = np.array([r["N"] for r in results if r["DRAM_Bytes"] > 0])
theory_Bs = Ns_fit * Ns_fit * 4  # 理论: N² * 4 bytes (读上三角 + 读x + 写x)
k = np.sum(Bs * theory_Bs) / np.sum(theory_Bs * theory_Bs)
print(f"\n  CPU TRSV DRAM = {k:.2f} * (N²*4)")
