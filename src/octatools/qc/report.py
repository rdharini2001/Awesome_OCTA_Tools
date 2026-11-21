"""The result object returned by the quality module."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class QCReport:
    grade: str                         # "good", "acceptable" or "reject"
    score: float                       # 0..1, weighted geometric mean of sub-scores
    metrics: dict[str, float]
    subscores: dict[str, float]
    flags: list[str] = field(default_factory=list)
    pixel_um: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.grade != "reject"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=kw.pop("indent", 2), **kw)

    def flat(self) -> dict:
        row = {"grade": self.grade, "score": round(self.score, 4), "flags": ";".join(self.flags)}
        row.update({f"m_{k}": v for k, v in self.metrics.items()})
        row.update({f"s_{k}": round(v, 4) for k, v in self.subscores.items()})
        return row


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """Write a list of flat dicts (e.g. {'file': ..., **report.flat()}) to CSV."""
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
