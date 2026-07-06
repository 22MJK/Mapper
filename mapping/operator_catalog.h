#pragma once

#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace mapping {

inline std::string normalize_operator_name(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return value;
}

// Canonical operator names accepted by the mapper. Operator parsing must go
// through this catalog before any cost or parallelism logic sees a task.
inline const std::unordered_set<std::string>& operator_profile_names() {
    static const std::unordered_set<std::string> values = {
        "scal",
        "noop",
        "copy",
        "axpy",
        "dot",
        "nrm2",
        "spmv",
        "sptrsv",
        "mv",
        "trsv",
        "trsm",
        "gemm",
        "spgemm",
        "transpose",
        "potrf",
        "geqrf",
        "assemble",
        "symbolic",
        "order",
        "etree",
        "colcount",
        "postorder",
        "supernode_partition",
        "send",
        "recv",
        "allreduce",
        "getrf",
        "tstrf",
        "gessm",
        "ssssm",
    };
    return values;
}

inline const std::unordered_set<std::string>& supported_operator_subtypes() {
    return operator_profile_names();
}

inline std::string canonical_operator_profile_name(std::string value) {
    value = normalize_operator_name(std::move(value));
    if (operator_profile_names().count(value) > 0) {
        return value;
    }
    if (value == "2_norm" || value == "2norm" || value == "l2_norm" || value == "norm" || value == "norm2") {
        return "nrm2";
    }
    if (value == "scale" || value == "scalar" || value == "scalar_add" || value == "scalar_div" ||
        value == "scalar_mul") {
        return "scal";
    }
    if (value == "gemv" || value == "matvec") {
        return "mv";
    }
    if (value == "triangular_solve") {
        return "trsv";
    }
    if (value == "sp_trsv" || value == "sparse_trsv" ||
        value == "sp_triangular_solve" ||
        value == "sparse_triangular_solve") {
        return "sptrsv";
    }
    if (value == "matmul" || value == "mm" || value == "bmm" || value == "gemm_update" ||
        value == "syrk_gemm" || value == "schur_update" || value == "update") {
        return "gemm";
    }
    if (value == "sp_gemm" || value == "sparse_gemm") {
        return "spgemm";
    }
    if (value == "supernode_factor" || value == "factor") {
        return "potrf";
    }
    if (value == "amd_ordering" || value == "colamd_ordering" || value == "metis_ordering" ||
        value == "natural_ordering" || value == "nesdis_ordering" || value == "ordering" ||
        value == "symbolic_order") {
        return "order";
    }
    if (value == "symbolic_etree" || value == "elimination_tree") {
        return "etree";
    }
    if (value == "column_counts") {
        return "colcount";
    }
    if (value == "symbolic_pattern" || value == "symbolic_dep_build" || value == "symbolic_supernode_detect") {
        return "symbolic";
    }
    if (value == "supernode_relax") {
        return "supernode_partition";
    }
    if (value == "comm_send_node") {
        return "send";
    }
    if (value == "comm_recv_node") {
        return "recv";
    }
    if (value == "comm_coll_node" || value == "all_reduce" || value == "allgather" || value == "all_gather" ||
        value == "reducescatter" || value == "reduce_scatter" || value == "broadcast" || value == "alltoall" ||
        value == "all_to_all") {
        return "allreduce";
    }
    return value;
}

inline std::string canonical_operator_subtype(std::string value) {
    return canonical_operator_profile_name(std::move(value));
}

inline bool is_operator_profile_name(const std::string& value) {
    return operator_profile_names().count(canonical_operator_profile_name(value)) > 0;
}

inline bool is_supported_operator_subtype(const std::string& value) {
    return is_operator_profile_name(value);
}

inline std::string require_operator_profile_name(const std::string& value) {
    const auto canonical = canonical_operator_profile_name(value);
    if (operator_profile_names().count(canonical) == 0) {
        throw std::runtime_error("Unsupported operator '" + value +
                                 "': mapper operator parsing is restricted to supported operator names");
    }
    return canonical;
}

}  // namespace mapping
