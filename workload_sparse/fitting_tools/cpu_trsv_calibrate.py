"""
cpu_trsv_calibrate.py — CPU TRSV DRAM 标定 (基于 VTune GUI 实测)
来源: VTune GUI Memory Access, N=4096 → DRAM=1.765 GB
"""

N_cal = 4096
DRAM_cal = 1.765e9       # 1.765 GB
theory_per_N2 = 4.0       # 4 bytes per element (fp32)
k = DRAM_cal / (N_cal * N_cal * theory_per_N2)

print("CPU TRSV DRAM 标定 (i9-13900HX, MSVC, 纯C++循环)")
print(f"  校准点: N={N_cal}, DRAM={DRAM_cal/1e9:.3f} GB")
print(f"  理论最小: N^2 * {theory_per_N2:.0f} B = {N_cal*N_cal*theory_per_N2/1e6:.1f} MB")
print(f"  实测倍数 k = {k:.2f}")
print()
print(f"  拟合公式: DRAM_CPU = {k:.1f} * N^2 * {theory_per_N2:.0f}  (bytes)")
print(f"           = {k*theory_per_N2:.1f} * N^2  (bytes)")
print()
print("  N        理论(MB)   预测DRAM(MB)   预测DRAM(GB)")
for N in [512, 1024, 2048, 4096, 6144, 8192, 10240]:
    t = N*N*theory_per_N2/1e6
    p = k * t
    print(f"  {N:<8}  {t:>10.1f}  {p:>12.1f}  {p/1000:>12.3f}")
