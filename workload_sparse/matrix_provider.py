import os
from typing import Optional, Tuple

# Optional: pip install ssgetpy
try:
    import ssgetpy

    HAS_SSGETPY = True
except ImportError:
    ssgetpy = None
    HAS_SSGETPY = False


def _filter_results_by_n(results, n_min: Optional[int], n_max: Optional[int]):
    if n_min is None and n_max is None:
        return results

    filtered = []
    for m in results:
        n = int(m.rows)
        if n_min is not None and n < n_min:
            continue
        if n_max is not None and n > n_max:
            continue
        filtered.append(m)

    return filtered


def fetch_suitesparse_spd(
    matrix_name: Optional[str],
    dest_dir: str,
    n_min: Optional[int],
    n_max: Optional[int],
    nnz_min: Optional[int],
    nnz_max: Optional[int],
) -> str:
    if not HAS_SSGETPY:
        raise RuntimeError("ssgetpy not available; install it or use --local-mtx.")
    os.makedirs(dest_dir, exist_ok=True)
    if matrix_name:
        results = ssgetpy.search(name=matrix_name, isspd=True)
    else:
        nzbounds = None
        if nnz_min is not None or nnz_max is not None:
            nzbounds = (nnz_min or 0, nnz_max or int(1e18))
        results = ssgetpy.search(isspd=True, nzbounds=nzbounds, limit=50)
    results = _filter_results_by_n(results, n_min, n_max)
    if not results:
        raise RuntimeError("No SPD matrix found in SuiteSparse query.")
    m = results[0]
    extracted, _ = m.download(destpath=dest_dir, extract=True)
    mtx = os.path.join(extracted, f"{m.name}.mtx")
    if not os.path.exists(mtx):
        raise FileNotFoundError(mtx)
    print(f"[Matrix] {m.name} shape={m.rows}x{m.cols} nnz={m.nnz}")
    return mtx


def matrix_provider(
    matrix_name: Optional[str],
    local_mtx: Optional[str],
    dest_dir: str,
    n_min: Optional[int],
    n_max: Optional[int],
    nnz_min: Optional[int],
    nnz_max: Optional[int],
) -> Tuple[str, str]:
    if local_mtx:
        if not os.path.exists(local_mtx):
            raise FileNotFoundError(local_mtx)
        name = os.path.basename(local_mtx).replace(".mtx", "")
        return name, local_mtx

    mtx = fetch_suitesparse_spd(matrix_name, dest_dir, n_min, n_max, nnz_min, nnz_max)
    name = os.path.basename(mtx).replace(".mtx", "")
    return name, mtx
