"""
fit_measured_data.py — 全 BLAS/LAPACK 算子联合标定 GPU 硬件极限

用法: python fit_measured_data.py
前提: 管理员运行 run_all_blas.bat + run_profiling.bat 生成 ncu_*.csv

输出:
  1. 各算子的 DRAM Bytes 测量值 vs README 理论值
  2. 联合拟合: T = T_launch + alpha*FLOPs + beta*DRAM_bytes
  3. 各算子 Bytes 经验公式
"""

import os, re, sys
import numpy as np
from scipy.optimize import curve_fit

FP32 = 4  # float32 bytes
IDX  = 4  # int32 bytes

# ═══════════════════════════════════════
# FLOPs 公式 (来源: README_ALL_FLOPs_Bytes.md)
# ═══════════════════════════════════════

def flops_trsv(N):     return 1.0 * N * N
def flops_gemv(M,N):   return 2.0 * M * N
def flops_symv(N):     return 2.0 * N * N
def flops_trmv(N):     return 1.0 * N * N
def flops_ger(M,N):    return 2.0 * M * N
def flops_syr(N):      return 1.0 * N * N
def flops_gemm(M,N,K): return 2.0 * M * N * K
def flops_symm(M,N):   return 2.0 * M * M * N
def flops_syrk(N,K):   return 1.0 * N * N * K
def flops_syr2k(N,K):  return 2.0 * N * N * K
def flops_trmm(M,N):   return 1.0 * M * N * N
def flops_trsm(M,N):   return 1.0 * M * N * N

# Bytes 公式 (README 理论值)
def bytes_trsv_theory(N):     return (N*N + 2*N)*FP32
def bytes_gemv_theory(M,N):   return (M*N + N + 2*M)*FP32
def bytes_symv_theory(N):     return (N*N/2 + 4*N)*FP32
def bytes_trmv_theory(N):     return (N*N + 2*N)*FP32
def bytes_ger_theory(M,N):    return (2*N + M + M*N)*FP32
def bytes_syr_theory(N):      return (N + N*N)*FP32
def bytes_gemm_theory(M,N,K): return (M*K + K*N + 2*M*N)*FP32
def bytes_symm_theory(M,N):   return (M*M + 3*M*N)*FP32
def bytes_syrk_theory(N,K):   return (N*K + N*N/2)*FP32
def bytes_syr2k_theory(N,K):  return (2*N*K + N*N/2)*FP32
def bytes_trmm_theory(M,N):   return (M*M + 2*M*N)*FP32
def bytes_trsm_theory(M,N):   return (M*M + 2*M*N)*FP32

# ═══════════════════════════════════════
# Helpers
# ═══════════════════════════════════════

def parse_ncu_csv(filepath: str, kernel_filter: str) -> dict:
    result = {}
    if not os.path.exists(filepath):
        return result
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    hdr = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("=="): continue
        if line.startswith('"ID"'): hdr = True; continue
        if not hdr: continue
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 15: continue
        # skip init/warmup kernels, take everything else
        kn = parts[4].lower()
        if "init" in kn or "warmup" in kn: continue
        if kernel_filter and kernel_filter.lower() not in kn: continue
        try: v = float(parts[14].replace(",",""))
        except ValueError: continue
        result[parts[12]] = result.get(parts[12], 0.0) + v
    return result

def model(X, t_launch, alpha, beta):
    return t_launch + alpha * X[0] + beta * X[1]


def main():
    DIR = os.path.dirname(os.path.abspath(__file__))
    all_F, all_B, all_T, all_L = [], [], [], []

    def add_point(label, flops, dram_bytes, time_s):
        if dram_bytes > 0 and time_s > 0:
            all_F.append(flops)
            all_B.append(dram_bytes)
            all_T.append(time_s)
            all_L.append(label)
            return True
        return False

    print("=" * 90)
    print(f"  {'算子':<8} {'规模':<20} {'FLOPs':>14} {'DRAM_ncu':>14} {'理论B':>14} {'时间us':>10} {'强度':>8}")
    print("-" * 90)

    # ═══ Level-1 (读 ncu 实测) ═══
    l1_ops_ncu = [
        ("scal", lambda n: n,       lambda n: 2*n*FP32),
        ("copy", lambda n: 0,       lambda n: 2*n*FP32),
        ("axpy", lambda n: 2*n,     lambda n: 3*n*FP32),
        ("dot",  lambda n: 2*n,     lambda n: 2*n*FP32+FP32),
        ("nrm2", lambda n: 2*n,     lambda n: n*FP32+FP32),
        ("rot",  lambda n: 6*n,     lambda n: 4*n*FP32),
        ("rotm", lambda n: 4*n,     lambda n: 4*n*FP32),
    ]
    for op_name, flop_fn, byte_fn in l1_ops_ncu:
        for N in [1<<18, 1<<20, 1<<22, 1<<23, 1<<24]:
            csv_path = os.path.join(DIR, f"ncu_L1_{op_name}_N{N}.csv")
            d = parse_ncu_csv(csv_path, "") if os.path.exists(csv_path) else {}
            flops = flop_fn(N); b_th = byte_fn(N)
            dram = d.get("dram__bytes.sum", 0); t_ns = d.get("gpu__time_duration.sum", 0)
            ai = flops/dram if dram>0 else 0
            tag = " [ncu]" if dram>0 else " [理论]"
            print(f"  {'L1_'+op_name:<8} N={N:<17} {flops:>14.0f} {dram:>14.0f}{tag} {b_th:>14.0f} {t_ns/1000:>10.2f} {ai:>8.2f}")
            add_point(f"L1_{op_name}_N{N}", flops, dram if dram>0 else b_th, t_ns*1e-9)

    # ═══ Level-2 (算子优先排序) ═══
    l2_sizes = [128,256,512,1024,2048,4096,8192]
    l2_ops = [
        ("gemv", flops_gemv, bytes_gemv_theory, "ncu_L2_GEMV", "gemv", True),
        ("ger",  flops_ger,  bytes_ger_theory,  "ncu_L2_GER",  "ger",  True),
        ("symv", flops_symv, bytes_symv_theory, "ncu_L2_SYMV", "symv", False),
        ("syr",  flops_syr,  bytes_syr_theory,  "ncu_L2_SYR",  "syr",  False),
        ("trmv", flops_trmv, bytes_trmv_theory, "ncu_L2_TRMV", "trmv", False),
    ]
    for op_name, flops_fn, bytes_fn, csv_prefix, kern_filter, uses_MN in l2_ops:
        for N in l2_sizes:
            csv_path = os.path.join(DIR, f"{csv_prefix}_N{N}.csv")
            d = parse_ncu_csv(csv_path, kern_filter)
            flops = flops_fn(N, N) if uses_MN else flops_fn(N)
            b_th  = bytes_fn(N, N) if uses_MN else bytes_fn(N)
            dram = d.get("dram__bytes.sum", 0)
            t_ns = d.get("gpu__time_duration.sum", 0)
            ai = flops/dram if dram>0 else 0
            tag = " [ncu]" if dram>0 else " [理论]"
            print(f"  {'L2_'+op_name:<8} N={N:<17} {flops:>14.0f} {dram:>14.0f}{tag} {b_th:>14.0f} {t_ns/1000:>10.2f} {ai:>8.2f}")
            add_point(f"L2_{op_name}_N{N}", flops, dram if dram>0 else b_th, t_ns*1e-9)

    # ═══ TRSV ═══
    for N in [128,256,512,1024,2048,4096,8192]:
        csv_path = os.path.join(DIR, f"ncu_TRSV_N{N}.csv")
        d = parse_ncu_csv(csv_path, "trsv")
        flops = flops_trsv(N); b_th = bytes_trsv_theory(N)
        dram = d.get("dram__bytes.sum", 0); t_ns = d.get("gpu__time_duration.sum", 0)
        ai = flops/dram if dram>0 else 0; tag = " [ncu]" if dram>0 else " [skip]"
        print(f"  {'TRSV':<8} N={N:<17} {flops:>14.0f} {dram:>14.0f}{tag} {b_th:>14.0f} {t_ns/1000:>10.2f} {ai:>8.2f}")
        add_point(f"TRSV_N{N}", flops, dram, t_ns*1e-9)

    # ═══ Level-3 (算子优先排序) ═══
    l3_sizes = [128,256,512,1024,2048,3072,4096]
    l3_ops = [
        ("gemm",  flops_gemm,  bytes_gemm_theory,  "ncu_L3_GEMM",  "gemm"),
        ("symm",  flops_symm,  bytes_symm_theory,  "ncu_L3_SYMM",  ""),
        ("syrk",  flops_syrk,  bytes_syrk_theory,  "ncu_L3_SYRK",  "gemm"),
        ("syr2k", flops_syr2k, bytes_syr2k_theory, "ncu_L3_SYR2K", "gemm"),
        ("trmm",  flops_trmm,  bytes_trmm_theory,  "ncu_L3_TRMM",  "trmm"),
    ]
    for op_name, flops_fn, bytes_fn, csv_prefix, kern_filter in l3_ops:
        for S in l3_sizes:
            csv_path = os.path.join(DIR, f"{csv_prefix}_N{S}.csv")
            d = parse_ncu_csv(csv_path, kern_filter)
            flops = flops_fn(S,S) if op_name!="gemm" else flops_fn(S,S,S)
            b_th  = bytes_fn(S,S) if op_name!="gemm" else bytes_fn(S,S,S)
            dram = d.get("dram__bytes.sum", 0); t_ns = d.get("gpu__time_duration.sum", 0)
            ai = flops/dram if dram>0 else 0; tag = " [ncu]" if dram>0 else " [理论]"
            print(f"  {'L3_'+op_name:<8} S={S:<17} {flops:>14.0f} {dram:>14.0f}{tag} {b_th:>14.0f} {t_ns/1000:>10.2f} {ai:>8.2f}")
            add_point(f"L3_{op_name}_S{S}", flops, dram if dram>0 else b_th, t_ns*1e-9)

    # ═══ Extra operators ═══
    extra_ops = {
        "GBMV":   {"csv":"ncu_GBMV","sizes":[256,512,1024,2048,4096,6144,8192],
                   "flops":lambda n: 2*(n//4+n//4+1)*n, "bytes":lambda n: ((n//4+n//4+1)*n+n+2*n)*FP32,
                   "kern":"gbmv","fmt":"N={:<17}"},
        "SPMV":   {"csv":"ncu_SPMV","sizes":[1024,2048,4096,8192,16384,32768,65536],
                   "flops":lambda n: 2*n*8, "bytes":lambda n: ((n+1)*IDX + n*8*(IDX+FP32) + 3*n*FP32),
                   "kern":"","fmt":"N={:<17}"},
        "SPTRSV": {"csv":"ncu_SPTRSV","sizes":[1024,2048,4096,8192,16384,32768,65536],
                   "flops":lambda n: 2*n*8, "bytes":lambda n: ((n+1)*IDX + n*8*(IDX+FP32) + 2*n*FP32),
                   "kern":"","fmt":"N={:<17}"},
        "SPGEMM": {"csv":"ncu_SPGEMM","sizes":[512,1024,2048,4096,6144,8192,10240],
                   "flops":lambda n: 2*n*16, "bytes":lambda n: (2*(n+1)*IDX + (n*4*2)*(IDX+FP32)),
                   "kern":"","fmt":"N={:<17}"},
        "TDENSE": {"csv":"ncu_TDENSE","sizes":[128,256,512,1024,2048,4096,6144],
                   "flops":lambda n: 0, "bytes":lambda n: 2*n*n*FP32,
                   "kern":"","fmt":"N={:<17}"},
        "TSPARSE":{"csv":"ncu_TSPARSE","sizes":[1024,2048,4096,8192,16384,32768,65536],
                   "flops":lambda n: 0, "bytes":lambda n: 2*((n+1)*IDX + n*8*(IDX+FP32)),
                   "kern":"","fmt":"N={:<17}"},
        "POTRF":  {"csv":"ncu_POTRF","sizes":[128,256,512,1024,2048,3072,4096],
                   "flops":lambda n: n**3/3, "bytes":lambda n: n*n*FP32,
                   "kern":"","fmt":"N={:<17}"},
        "GETRF":  {"csv":"ncu_GETRF","sizes":[128,256,512,1024,2048,3072,4096],
                   "flops":lambda n: 2*n**3/3, "bytes":lambda n: (n**2.5/1.0)*FP32,
                   "kern":"","fmt":"N={:<17}"},
        "GEQRF":  {"csv":"ncu_GEQRF","sizes":[128,256,512,1024,2048,3072,4096],
                   "flops":lambda n: 4*n**3/3, "bytes":lambda n: (n**2.6/10)*FP32,
                   "kern":"","fmt":"N={:<17}"},
    }
    for name, cfg in extra_ops.items():
        for N in cfg["sizes"]:
            csv_path = os.path.join(DIR, f"{cfg['csv']}_N{N}.csv")
            d = parse_ncu_csv(csv_path, cfg["kern"])
            flops = cfg["flops"](N); b_th = cfg["bytes"](N)
            dram = d.get("dram__bytes.sum", 0); t_ns = d.get("gpu__time_duration.sum", 0)
            ai = flops/dram if dram>0 else 0; tag = " [ncu]" if dram>0 else " [理论]"
            print(f"  {name:<8} {cfg['fmt'].format(N)} {flops:>14.0f} {dram:>14.0f}{tag} {b_th:>14.0f} {t_ns/1000:>10.2f} {ai:>8.2f}")
            add_point(f"{name}_N{N}", flops, dram if dram>0 else b_th, t_ns*1e-9)

    # ═══════════════════════════════════════
    # 联合拟合
    # ═══════════════════════════════════════
    if len(all_F) < 5:
        print(f"\n[WARN] 只有 {len(all_F)} 个有效数据点, 跳过拟合. 请先运行 run_all_blas.bat (管理员)")
        return

    F = np.array(all_F); B = np.array(all_B); T = np.array(all_T)
    sf, sb = np.mean(F), np.mean(B)
    Xn = np.vstack([F/sf, B/sb])
    popt, _ = curve_fit(model, Xn, T, bounds=(0, np.inf), maxfev=10000)
    t_launch, alpha, beta = popt[0], popt[1]/sf, popt[2]/sb

    Tp = np.array([model([F[i]/sf, B[i]/sb], *popt) for i in range(len(F))])
    res = T - Tp
    SSr, SSt = np.sum(res**2), np.sum((T-np.mean(T))**2)
    r2 = 1 - SSr/SSt if SSt>0 else 0

    print(f"\n{'='*90}")
    print(f"  联合拟合 ({len(F)} 个数据点, R^2={r2:.4f})")
    print(f"{'='*90}")
    print(f"  T_launch  : {t_launch*1e6:.2f} us")
    print(f"  Alpha     : {alpha:.4e} s/FLOP  → 算力 {1/alpha/1e9:.1f} GFLOPS")
    print(f"  Beta      : {beta:.4e} s/Byte   → 带宽 {1/beta/1e9:.1f} GB/s")
    print(f"{'='*90}")

    # ═══════════════════════════════════════
    # Bytes 经验公式 (对每个算子族)
    # ═══════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  DRAM Bytes 对比: ncu实测 vs README理论")
    print(f"{'='*90}")

    families = [
        ("SCAL", "ncu_L1_scal", "", [1<<18,1<<20,1<<22,1<<23,1<<24], "N"),
        ("COPY", "ncu_L1_copy", "", [1<<18,1<<20,1<<22,1<<23,1<<24], "N"),
        ("AXPY", "ncu_L1_axpy", "", [1<<18,1<<20,1<<22,1<<23,1<<24], "N"),
        ("DOT",  "ncu_L1_dot",  "", [1<<18,1<<20,1<<22,1<<23,1<<24], "N"),
        ("NRM2", "ncu_L1_nrm2", "", [1<<18,1<<20,1<<22,1<<23,1<<24], "N"),
        ("ROT",  "ncu_L1_rot",  "", [1<<18,1<<20,1<<22,1<<23,1<<24], "N"),
        ("ROTM", "ncu_L1_rotm", "", [1<<18,1<<20,1<<22,1<<23,1<<24], "N"),
        ("TRSV", "ncu_TRSV", "trsv", [128,256,512,1024,2048,4096,8192], "N"),
        ("GEMV", "ncu_L2_GEMV", "gemv", [128,256,512,1024,2048,4096,8192], "N"),
        ("SYMV", "ncu_L2_SYMV", "symv", [128,256,512,1024,2048,4096,8192], "N"),
        ("TRMV", "ncu_L2_TRMV", "trmv", [128,256,512,1024,2048,4096,8192], "N"),
        ("GER",  "ncu_L2_GER",  "ger",  [128,256,512,1024,2048,4096,8192], "N"),
        ("SYR",  "ncu_L2_SYR",  "syr",  [128,256,512,1024,2048,4096,8192], "N"),
        ("GEMM", "ncu_L3_GEMM", "gemm", [128,256,512,1024,2048,3072,4096], "S"),
        ("SYMM", "ncu_L3_SYMM", "", [128,256,512,1024,2048,3072,4096], "S"),
        ("SYRK", "ncu_L3_SYRK", "gemm", [128,256,512,1024,2048,3072,4096], "S"),
        ("SYR2K","ncu_L3_SYR2K","gemm",[128,256,512,1024,2048,3072,4096], "S"),
        ("TRMM", "ncu_L3_TRMM", "trmm", [128,256,512,1024,2048,3072,4096], "S"),
        ("TRSM", "ncu_TRSM",    "trsm",    [128,256,512,1024,2048,3072,4096], "S"),
        ("GBMV", "ncu_GBMV", "gbmv", [256,512,1024,2048,4096,6144,8192], "N"),
        ("SPMV", "ncu_SPMV", "", [1024,2048,4096,8192,16384,32768,65536], "N"),
        ("SPTRSV","ncu_SPTRSV","",[1024,2048,4096,8192,16384,32768,65536], "N"),
        ("SPGEMM","ncu_SPGEMM","",[512,1024,2048,4096,6144,8192,10240], "N"),
        ("TDENSE","ncu_TDENSE","",[128,256,512,1024,2048,4096,6144], "N"),
        ("TSPARSE","ncu_TSPARSE","",[1024,2048,4096,8192,16384,32768,65536],"N"),
        ("POTRF","ncu_POTRF","",[128,256,512,1024,2048,3072,4096], "N"),
        ("GETRF","ncu_GETRF","",[128,256,512,1024,2048,3072,4096], "N"),
        ("GEQRF","ncu_GEQRF","",[128,256,512,1024,2048,3072,4096], "N"),
    ]

    # 理论Bytes映射
    theory_bytes = {
        "SCAL":  lambda n: 2*int(n)*FP32,
        "COPY":  lambda n: 2*int(n)*FP32,
        "AXPY":  lambda n: 3*int(n)*FP32,
        "DOT":   lambda n: 2*int(n)*FP32+FP32,
        "NRM2":  lambda n: int(n)*FP32+FP32,
        "ROT":   lambda n: 4*int(n)*FP32,
        "ROTM":  lambda n: 4*int(n)*FP32,
        "TRSV":  lambda n: bytes_trsv_theory(int(n)),
        "GEMV":  lambda n: bytes_gemv_theory(int(n), int(n)),
        "SYMV":  lambda n: bytes_symv_theory(int(n)),
        "TRMV":  lambda n: bytes_trmv_theory(int(n)),
        "GER":   lambda n: bytes_ger_theory(int(n), int(n)),
        "SYR":   lambda n: bytes_syr_theory(int(n)),
        "GEMM":  lambda n: bytes_gemm_theory(int(n), int(n), int(n)),
        "SYMM":  lambda n: bytes_symm_theory(int(n), int(n)),
        "SYRK":  lambda n: bytes_syrk_theory(int(n), int(n)),
        "SYR2K": lambda n: bytes_syr2k_theory(int(n), int(n)),
        "TRMM":  lambda n: bytes_trmm_theory(int(n), int(n)),
        "TRSM":  lambda n: bytes_trsm_theory(int(n), int(n)),
        "GBMV":  lambda n: ((n//4+n//4+1)*n+n+2*n)*FP32,
        "SPMV":  lambda n: ((n+1)*IDX+n*8*(IDX+FP32)+3*n*FP32),
        "SPTRSV":lambda n: ((n+1)*IDX+n*8*(IDX+FP32)+2*n*FP32),
        "SPGEMM":lambda n: (2*(n+1)*IDX+(n*4*2)*(IDX+FP32)),
        "TDENSE":lambda n: 2*n*n*FP32,
        "TSPARSE":lambda n:2*((n+1)*IDX+n*8*(IDX+FP32)),
        "POTRF": lambda n: n*n*FP32,
        "GETRF": lambda n: (int(n)**2.5/1.0)*FP32,
        "GEQRF": lambda n: (int(n)**2.6/10)*FP32,
    }

    for name, prefix, kf, sizes, unit in families:
        vals = []
        theory_vals = []
        for n in sizes:
            p = os.path.join(DIR, f"{prefix}_N{n}.csv")
            d = parse_ncu_csv(p, kf)
            vals.append(d.get("dram__bytes.sum", 0))
            theory_vals.append(theory_bytes.get(name, lambda x: 0)(n))
        vals = np.array(vals)
        theory_vals = np.array(theory_vals)
        if not np.any(vals > 0): continue

        sizes_arr = np.array(sizes)
        mask = vals > 0
        if mask.sum() < 2: continue
        t_eff = theory_vals[mask]; v_eff = vals[mask]

        # 搜索最优 k: 最小化平均绝对百分比误差 (MAPE)
        best_k, best_mape = 0, 1e99
        for k in np.linspace(0.5, 30, 600):
            mape = np.mean(np.abs((k * t_eff - v_eff) / v_eff))
            if mape < best_mape:
                best_mape, best_k = mape, k
        pred = best_k * theory_vals

        print(f"\n  [{name}] DRAM = {best_k:.2f} * THEORY  (MAPE={best_mape*100:.1f}%)")
        print(f"  {'规模':>10} {'ncu实测':>14} {'拟合(k*理论)':>14} {'误差%':>8} {'实测/理论':>8} {'实测/拟合':>8}")
        for i, n in enumerate(sizes):
            if vals[i] > 0:
                err = (pred[i]-vals[i])/vals[i]*100
                ratio = vals[i]/theory_vals[i] if theory_vals[i]>0 else 0
                fit_ratio = vals[i]/pred[i] if pred[i]>0 else 0
                print(f"  {n:>10} {vals[i]:>14.0f} {pred[i]:>14.0f} {err:>7.2f}% {ratio:>8.1f}x {fit_ratio:>8.2f}x")

    print(f"\n{'='*90}")

if __name__ == "__main__":
    main()

