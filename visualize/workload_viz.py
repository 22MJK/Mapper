#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def load_workload(path: Path):
    with path.open() as f:
        return json.load(f)


def make_label(task, simple=False):
    tid = task.get("id")
    name = task.get("name", "")
    ttype = task.get("type", "")
    subtype = task.get("subtype", "")
    if simple:
        return f"{name}" if name else f"{tid}"
    if subtype:
        return f"{tid}: {name}\\n{ttype}/{subtype}"
    return f"{tid}: {name}\\n{ttype}"


def to_dot(data, include_bytes=True, simple=False, compact=False, big_nodes=False):
    tasks = data.get("tasks", [])
    tensors = data.get("tensors", [])
    tensor_map = {t.get("id"): t for t in tensors if isinstance(t, dict) and t.get("id") is not None}
    node_fontsize = 18 if big_nodes else 14
    edge_fontsize = 14 if big_nodes else 12
    node_margin = "0.28,0.18" if big_nodes else "0.18,0.10"
    lines = [
        "digraph workload {",
        "  rankdir=LR;",
        "  splines=true;",
        "  concentrate=true;",
        "  nodesep=0.6;",
        "  ranksep=0.8;",
        f"  node [shape=box, style=rounded, fontname=\"Helvetica\", fontsize={node_fontsize}, margin=\"{node_margin}\"];",
        f"  edge [fontname=\"Helvetica\", fontsize={edge_fontsize}, color=\"#444444\"];",
    ]

    # Optional iteration I/O nodes.
    iter_inputs = set(data.get("iteration_inputs", []))
    iter_outputs = set(data.get("iteration_outputs", []))
    if iter_inputs and not compact:
        lines.append("  IterIn [shape=box, style=\"dashed,rounded\", label=\"iteration_inputs\"];")
    if iter_outputs and not compact:
        lines.append("  IterOut [shape=box, style=\"dashed,rounded\", label=\"iteration_outputs\"];")

    for t in tasks:
        label = make_label(t, simple=simple).replace('"', '')
        lines.append(f"  T{t['id']} [label=\"{label}\"]; ")

    def edge_label(tensor_id, tensor, inp):
        parts = [tensor_id]
        if not simple:
            access = None
            if isinstance(inp, dict):
                access = inp.get("access") or inp.get("access_pattern")
            if access:
                parts.append(str(access))
            if include_bytes and isinstance(tensor, dict):
                bytes_val = tensor.get("size_bytes", tensor.get("bytes", 0))
                if bytes_val:
                    parts.append(f"{bytes_val}B")
        return " | ".join(parts) if parts else ""

    edge_list = []
    for t in tasks:
        dst = t.get("id")
        inputs = t.get("inputs", [])
        for inp in inputs:
            if not isinstance(inp, dict):
                continue
            tensor_id = inp.get("tensor")
            if tensor_id is None:
                continue
            tensor = tensor_map.get(tensor_id)
            producer = tensor.get("producer") if tensor else None
            label = edge_label(tensor_id, tensor, inp)
            if producer is None:
                if tensor_id in iter_inputs and not compact:
                    edge_list.append(("IterIn", f"T{dst}", label))
                continue
            edge_list.append((f"T{producer}", f"T{dst}", label))

    for tensor_id in iter_outputs:
        tensor = tensor_map.get(tensor_id)
        if not tensor:
            continue
        producer = tensor.get("producer")
        if producer is None:
            continue
        label = edge_label(tensor_id, tensor, {})
        if not compact:
            edge_list.append((f"T{producer}", "IterOut", label))

    if compact:
        seen = set()
        for src, dst, _label in edge_list:
            if src == "IterIn" or dst == "IterOut":
                continue
            key = (src, dst)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  {src} -> {dst}; ")
    else:
        for src, dst, label in edge_list:
            if label:
                lines.append(f"  {src} -> {dst} [label=\"{label}\"]; ")
            else:
                lines.append(f"  {src} -> {dst}; ")

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_png_from_dot(dot_text: str, png_path: Path) -> int:
    dot = shutil.which("dot")
    if not dot:
        print("error: Graphviz 'dot' not found; install graphviz.", file=sys.stderr)
        return 2
    proc = subprocess.run([dot, "-Tpng", "-o", str(png_path)], input=dot_text.encode("utf-8"))
    return proc.returncode


def write_svg_from_dot(dot_text: str, svg_path: Path) -> int:
    dot = shutil.which("dot")
    if not dot:
        print("error: Graphviz 'dot' not found; install graphviz.", file=sys.stderr)
        return 2
    proc = subprocess.run([dot, "-Tsvg", "-o", str(svg_path)], input=dot_text.encode("utf-8"))
    return proc.returncode


def parse_args():
    p = argparse.ArgumentParser(description="Visualize workload.json as Graphviz DOT/SVG/PNG.")
    p.add_argument("--input", required=True, help="Path to workload.json")
    p.add_argument("--output", help="Output file; defaults to workload.svg")
    p.add_argument("--png", help="Write PNG to this path (requires dot)")
    p.add_argument("--no-bytes", action="store_true", help="Do not label edges with size_bytes")
    p.add_argument("--simple", action="store_true", help="Simplify node/edge labels for readability")
    p.add_argument("--compact", action="store_true", help="Aggressively reduce edges (deduplicate, hide iteration I/O)")
    p.add_argument("--big-nodes", action="store_true", help="Increase node size and font for readability")
    return p.parse_args()


def main():
    args = parse_args()
    data = load_workload(Path(args.input))
    include_bytes = not args.no_bytes
    output = to_dot(
        data,
        include_bytes=include_bytes,
        simple=args.simple,
        compact=args.compact,
        big_nodes=args.big_nodes,
    )

    if args.png:
        sys.exit(write_png_from_dot(output, Path(args.png)))

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".dot":
            out_path.write_text(output)
            return
        if out_path.suffix == ".png":
            sys.exit(write_png_from_dot(output, out_path))
        sys.exit(write_svg_from_dot(output, out_path))

    sys.exit(write_svg_from_dot(output, Path("workload.svg")))


if __name__ == "__main__":
    main()
