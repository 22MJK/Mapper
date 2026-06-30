import json
from typing import Any, Dict


def write_workload(path: str, workload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workload, f, indent=2, ensure_ascii=False)
