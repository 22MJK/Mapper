# BLAS / LAPACK / CHOLMOD 算子 compute_FLOPs 与 Bytes 汇总表

> 来源：cuBLAS Library v13.2 (NVIDIA, 2026) + Netlib 参考源码 + BLAS Technical Forum Standard (2002) + CHOLMOD (Davis, 2008)
> 
> **符号约定：** `S` = 浮点数字节数 (fp64→8, fp32→4)，`I` = 索引字节数 (int32→4, int64→8)

---

## Level-1 BLAS（向量操作）

| 算子       | 数学定义                                                                                                                | 输入                               | 输出                   | compute_FLOPs | Bytes           | BytesGPU拟合    |
|:-------- |:------------------------------------------------------------------------------------------------------------------- |:-------------------------------- |:-------------------- |:------------- |:--------------- |:------------- |
| **scal** | $x \leftarrow \alpha \cdot x$ | $x \in \mathbb{R}^n$, $\alpha$ | $x$ (覆盖) | $n$ | $2n\cdot S$ | **0.50 × Bytes** |
| **copy** | $y \leftarrow x$ | $x \in \mathbb{R}^n$ | $y \in \mathbb{R}^n$ | $0$ | $2n\cdot S$ | **0.50 × Bytes** |
| **axpy** | $y \leftarrow \alpha \cdot x + y$ | $x,y \in \mathbb{R}^n$, $\alpha$ | $y$ (覆盖) | $2n$ | $3n\cdot S$ | **0.70 × Bytes** |
| **dot**  | $\alpha \leftarrow x^T y$ | $x,y \in \mathbb{R}^n$ | $\alpha$ (标量) | $2n$ | $2n\cdot S + S$ | **0.99 × Bytes** |
| **nrm2** | $\alpha \leftarrow \|x\|_2$ | $x \in \mathbb{R}^n$ | $\alpha$ (标量) | $2n$ | $n\cdot S + S$ | **0.99 × Bytes** |
| **rot**  | Givens 旋转 | $x,y\in\mathbb{R}^n,c,s$ | $x,y$ (覆盖) | $6n$ | $4n\cdot S$ | **0.55 × Bytes** |
| **rotm** | 修正 Givens | $x,y\in\mathbb{R}^n$, param | $x,y$ (覆盖) | $4n$ | $4n\cdot S$ | **0.55 × Bytes** |

---

## Level-2 BLAS（矩阵-向量操作）

| 算子       | 数学定义                                                              | 输入                                                                                   | 输出                 | compute_FLOPs              | Bytes                                                   | BytesGPU拟合       |
|:-------- |:----------------------------------------------------------------- |:------------------------------------------------------------------------------------ |:------------------ |:-------------------------- |:------------------------------------------------------- |:---------------- |
| **gemv** | $y \leftarrow \alpha \cdot \text{op}(A)\cdot x + \beta \cdot y$   | $A\in\mathbb{R}^{m\times n}$, $x\in\mathbb{R}^n$, $y\in\mathbb{R}^m$, $\alpha,\beta$ | $y$ (覆盖)           | $2mn$（$\beta\neq0$ 时 $+m$） | $(mn + n + 2m)\cdot S$                                  | **2.03 × Bytes** |
| **gbmv** | $y \leftarrow \alpha\cdot\text{op}(A)\cdot x + \beta\cdot y$ (带状) | $A$ 带状 $m\times n$, $kl,ku$, $x,y,\alpha,\beta$                                      | $y$ (覆盖)           | $2(kl+ku+1)\cdot\min(m,n)$ | $(kl+ku+1)\cdot\min(m,n)\cdot S + n\cdot S + 2m\cdot S$ | **1.78 × Bytes** |
| **symv** | $y \leftarrow \alpha\cdot A\cdot x + \beta\cdot y$ (对称)           | $A\in\mathbb{R}^{n\times n}$ 对称, $x,y\in\mathbb{R}^n,\alpha,\beta$                   | $y$ (覆盖)           | $2n^2$                     | $(n^2/2 + 2n + 2n)\cdot S$                              | **2.03 × Bytes** |
| **trmv** | $x \leftarrow \text{op}(T)\cdot x$ (三角矩阵-向量)                      | $T\in\mathbb{R}^{n\times n}$ 三角, $x\in\mathbb{R}^n$                                  | $x$ (覆盖)           | $n^2$                      | $(n^2 + 2n)\cdot S$                                     | **1.04 × Bytes** |
| **trsv** | 解 $\text{op}(T)\cdot x = b$ (稠密三角求解)                              | $T\in\mathbb{R}^{n\times n}$ 三角, $b\in\mathbb{R}^n$                                  | $x\in\mathbb{R}^n$ | $n^2$                      | $(n^2 + 2n)\cdot S$                                     | **0.50 × Bytes** |
| **ger**  | $A \leftarrow \alpha\cdot x\cdot y^T + A$                         | $x\in\mathbb{R}^m$, $y\in\mathbb{R}^n$, $A\in\mathbb{R}^{m\times n}$, $\alpha$       | $A$ (覆盖)           | $2mn$                      | $(2n+m+mn)\cdot S$                                      | **2.03 × Bytes** |
| **syr**  | $A \leftarrow \alpha\cdot x\cdot x^T + A$ (对称秩-1)                 | $x\in\mathbb{R}^n$, $A\in\mathbb{R}^{n\times n}$ 对称, $\alpha$                        | $A$ (覆盖)           | $n^2$                      | $(n + n^2/2 + n^2/2)\cdot S$                            | **1.04 × Bytes** |

> **op(A)** 可为 $A$（不转置）、$A^T$（转置）、$A^H$（共轭转置），不影响 FLOPs。

---

## Level-2 BLAS — 稀疏版本（cuSPARSE / Sparse BLAS Standard）

| 算子         | 数学定义                                                     | 输入                                                                                           | 输出                 | compute_FLOPs   | Bytes（CSR/CSC格式）                                      | BytesGPU拟合       |
|:---------- |:-------------------------------------------------------- |:-------------------------------------------------------------------------------------------- |:------------------ |:--------------- |:----------------------------------------------------- |:---------------- |
| **SpMV**   | $y \leftarrow \alpha\cdot A\cdot x + \beta\cdot y$ (CSR) | 稀疏 $A\in\mathbb{R}^{m\times n}$(CSR), $x\in\mathbb{R}^n$, $y\in\mathbb{R}^m$, $\alpha,\beta$ | $y$ (覆盖)           | $2\cdot nnz$    | $(m+1)\cdot I + nnz\cdot(I+S) + n\cdot S + 2m\cdot S$ | **1.93 × Bytes** |
| **稀疏trsv** | 解 $L\cdot x = b$ (CSR)                                   | 稀疏 $L\in\mathbb{R}^{n\times n}$(CSR), $b\in\mathbb{R}^n$                                     | $x\in\mathbb{R}^n$ | $2\cdot nnz(L)$ | $(n+1)\cdot I + nnz\cdot(I+S) + 2n\cdot S$            | **2.86 × Bytes** |

---

## Level-3 BLAS（矩阵-矩阵操作）

| 算子         | 数学定义                                                                   | 输入                                                                                                       | 输出                                   | compute_FLOPs                             | Bytes                                             | BytesGPU拟合       |
|:---------- |:---------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------------------- |:------------------------------------ |:----------------------------------------- |:------------------------------------------------- |:---------------- |
| **gemm**   | $C \leftarrow \alpha\cdot\text{op}(A)\cdot\text{op}(B) + \beta\cdot C$ | $A\in\mathbb{R}^{m\times k}$, $B\in\mathbb{R}^{k\times n}$, $C\in\mathbb{R}^{m\times n}$, $\alpha,\beta$ | $C$ (覆盖)                             | **$2mnk$**                                | $(mk + kn + 2mn)\cdot S$                          | **1.14 × Bytes** |
| **symm**   | $C\leftarrow\alpha A B + \beta C$ 或 $\alpha B A + \beta C$ (对称)        | $A\in\mathbb{R}^{m\times m}$对称, $B,C\in\mathbb{R}^{m\times n}$, $\alpha,\beta$                           | $C$ (覆盖)                             | $2m^2 n$                                  | $(m^2 + mn + 2mn)\cdot S$                         | **1.63 × Bytes** |
| **syrk**   | $C\leftarrow\alpha A A^T+\beta C$ (对称秩-k)                              | $A\in\mathbb{R}^{n\times k}$, $C\in\mathbb{R}^{n\times n}$对称, $\alpha,\beta$                             | $C$下三角(覆盖)                           | $n^2 k$                                   | $(nk + n^2/2)\cdot S$                             | **1.83 × Bytes** |
| **syr2k**  | $C\leftarrow\alpha(AB^T+BA^T)+\beta C$ (对称秩-2k)                        | $A,B\in\mathbb{R}^{n\times k}$, $C\in\mathbb{R}^{n\times n}$对称, $\alpha,\beta$                           | $C$下三角(覆盖)                           | $2n^2 k$                                  | $(2nk + n^2/2)\cdot S$                            | **5.52 × Bytes** |
| **trmm**   | $B\leftarrow\alpha\text{op}(T)B$ 或 $\alpha B\text{op}(T)$              | $T\in\mathbb{R}^{n\times n}$三角, $B\in\mathbb{R}^{n\times m}$, $\alpha$                                   | $B$ (覆盖)                             | $mn^2$                                    | $(n^2 + mn + mn)\cdot S$                          | **1.09 × Bytes** |
| **trsm**   | 解 $\text{op}(T)X = \alpha B$ 或 $X\text{op}(T)=\alpha B$                | $T\in\mathbb{R}^{n\times n}$三角, $B\in\mathbb{R}^{n\times m}$, $\alpha$                                   | $X\in\mathbb{R}^{n\times m}$         | $mn^2$                                    | $(n^2 + 2nm)\cdot S$                              | **0.75 × Bytes** |
| **spgemm** | $C \leftarrow A \times B$ (CSR×CSR)                                    | 稀疏 $A\in\mathbb{R}^{m\times k}$(CSR), 稀疏 $B\in\mathbb{R}^{k\times n}$(CSR)                               | 稀疏 $C\in\mathbb{R}^{m\times n}$(CSR) | $2\cdot\sum_{A(i,k)\neq0}nnz(B_{row\_k})$ | $2[(n+1)\cdot I] + (nnz_a+nnz_b+nnz_c)\cdot(I+S)$ | **4.78 × Bytes** |

---

## 转置

| 算子                 | 数学定义               | 输入                                   | 输出                           | compute_FLOPs | Bytes                             | BytesGPU拟合       |
|:------------------ |:------------------ |:------------------------------------ |:---------------------------- |:------------- |:--------------------------------- |:---------------- |
| **transpose** (稠密) | $B \leftarrow A^T$ | $A\in\mathbb{R}^{m\times n}$         | $B\in\mathbb{R}^{n\times m}$ | $0$           | $2mn\cdot S$                      | **1.04 × Bytes** |
| **transpose** (稀疏) | CSR $\to$ CSC      | 稀疏 $A\in\mathbb{R}^{n\times n}$(CSR) | 稀疏 $A^T$(CSC)                | $0$           | $2[(n+1)\cdot I + nnz\cdot(I+S)]$ | **2.67 × Bytes** |

---

## LAPACK 分解算子

| 算子                                    | 数学定义                         | 输入                                      | 输出             | compute_FLOPs         | Bytes          | BytesGPU拟合       |
|:------------------------------------- |:---------------------------- |:--------------------------------------- |:-------------- |:--------------------- |:-------------- |:---------------- |
| **potrf**                             | $A = L\cdot L^T$ (Cholesky)  | $A\in\mathbb{R}^{n\times n}$ SPD, 仅下三角  | $L$ (覆盖A下三角)   | $\frac{1}{3}n^3$      | $n^2\cdot S$   | **1.19 × Bytes** |
| **getrf** | $A = P\cdot L\cdot U$ (LU分解) | $A\in\mathbb{R}^{m\times n}$            | $L,U,P$ (原位覆盖) | $\frac{2}{3}n^3$ (方阵) | $2mn\cdot S$ * | **0.50 × Bytes** |
| **geqrf**                             | $A = Q\cdot R$ (QR分解)        | $A\in\mathbb{R}^{m\times n}$ ($m\ge n$) | $Q,R$          | $2n^2(m-n/3)$         | $2mn\cdot S$ * | **3.36 × Bytes** |

> * GETRF/GEQRF 的 README Bytes 公式仅为理论下界($O(N^2)$)。实际分块算法反复读写尾矩阵，DRAM 呈 $O(N^{2.5})$ (GETRF) / $O(N^{2.6})$ (GEQRF)。拟合系数基于经验幂律公式。其余算子 Bytes 严格按 README。

---

## CHOLMOD 超节点算子

| 算子        | 数学定义                  | 输入                                                       | 输出                   | compute_FLOPs                    | Bytes                                                | BytesGPU拟合                |
|:--------- |:--------------------- |:-------------------------------------------------------- |:-------------------- |:-------------------------------- |:---------------------------------------------------- |:------------------------- |
| **tstrf** | POTRF(对角块) + TRSM(面板) | $F\in\mathbb{R}^{(ns+nu)\times(ns+nu)}$                  | $L_{11}$ + panel     | $\frac{1}{3}ns^3 + nu\cdot ns^2$ | $(ns+nu)^2\cdot S + ns^2\cdot S + nu\cdot ns\cdot S$ | 1.19×POTRF + 0.75×TRSM 组合 |
| **gessm** | 子节点面板 SYRK 更新         | `child_panel` $\in\mathbb{R}^{ov\times ns}$              | 更新矩阵 $U$             | $ov\cdot ns^2$                   | $(ov\cdot ns + ov^2)\cdot S$                         | **1.83 × Bytes**          |
| **ssssm** | 稀疏前代 + 回代             | 稀疏 $L\in\mathbb{R}^{n\times n}$(CSC), $b\in\mathbb{R}^n$ | 解 $x\in\mathbb{R}^n$ | $4\cdot nnz(L)$                  | $2[(n+1)\cdot I + nnz\cdot(I+S)] + 3n\cdot S$        | 2×0.50×TRSV Bytes         |

> **实测环境**: RTX 4060 Laptop (SM 8.9), CUDA 12.6, Nsight Compute 2024.3.0; BytesGPU = k × Bytes(理论), k 由最小化平均百分比误差 (MAPE) 拟合 7 组不同规模的 ncu 实测 DRAM 得到。
