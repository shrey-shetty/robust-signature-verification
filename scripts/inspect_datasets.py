"""Dataset integrity diagnostics for signature verification datasets.

Supports two layouts:
  A) Per-writer folders (GPDS Synthetic, BHSig260 Bengali/Hindi, CEDAR):
     <root>/<writer_id>/<files>
  B) Flat org/forg folders (institutional dataset):
     <root>/full_org/<genuine files>, <root>/full_forg/<forgery files>
     with the writer ID embedded in each filename.

Checks:
  1. Writer count.
  2. Per-writer genuine/forgery counts; deviations from the modal
     (most common) count are flagged. No expected counts are hard-coded.
  3. Layout A only: filename writer-ID vs. containing folder consistency.
  4. Files not matching the expected naming pattern ("unrecognized").

NOT checked: image readability/corruption (files are never opened),
duplicate content, dimensions. Those belong in the preprocessing pass.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\inspect_datasets.py
    .\\.venv\\Scripts\\python.exe scripts\\inspect_datasets.py --csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"


@dataclass
class WriterStats:
    genuine: int = 0
    forgery: int = 0
    mismatched_files: list[str] = field(default_factory=list)
    unrecognized_files: list[str] = field(default_factory=list)


@dataclass
class PerWriterFolderSpec:
    """Layout A: one folder per writer."""
    name: str
    root: Path
    genuine_re: re.Pattern   # must expose group 'writer'
    forgery_re: re.Pattern   # must expose group 'writer'
    folder_id_re: re.Pattern = re.compile(r"^(?P<writer>\d+)$")


@dataclass
class FlatFoldersSpec:
    """Layout B: all writers mixed in a genuine folder and a forgery folder."""
    name: str
    genuine_dir: Path
    forgery_dir: Path
    genuine_re: re.Pattern   # must expose group 'writer'
    forgery_re: re.Pattern   # must expose group 'writer'


DATASETS: list[PerWriterFolderSpec | FlatFoldersSpec] = [
    PerWriterFolderSpec(
        name="GPDS Synthetic 4000",
        root=DATA_RAW / "SignatureGPDSSyntheticOffLine4000" / "firmasSINTESISmanuscritas",
        # forgery tested before genuine in code: 'cf-' would otherwise be
        # shadowed if a looser genuine pattern were used.
        genuine_re=re.compile(r"^c-(?P<writer>\d+)-\d+\.jpg$", re.IGNORECASE),
        forgery_re=re.compile(r"^cf-(?P<writer>\d+)-\d+\.jpg$", re.IGNORECASE),
    ),
    PerWriterFolderSpec(
        name="BHSig260 Bengali",
        root=DATA_RAW / "BHSig260-Bengali",
        genuine_re=re.compile(r"^B-S-(?P<writer>\d+)-G-\d+\.tif$", re.IGNORECASE),
        forgery_re=re.compile(r"^B-S-(?P<writer>\d+)-F-\d+\.tif$", re.IGNORECASE),
    ),
    PerWriterFolderSpec(
        name="BHSig260 Hindi",
        root=DATA_RAW / "BHSig260-Hindi",
        genuine_re=re.compile(r"^H-S-(?P<writer>\d+)-G-\d+\.tif$", re.IGNORECASE),
        forgery_re=re.compile(r"^H-S-(?P<writer>\d+)-F-\d+\.tif$", re.IGNORECASE),
    ),
    PerWriterFolderSpec(
        name="CEDAR",
        root=DATA_RAW / "CEDAR",
        # NOTE: 'original_' genuine pattern is inferred from the standard
        # CEDAR convention; only forgery filenames were directly observed.
        # If genuine files use a different prefix they will show up as
        # "unrecognized" below — that is the signal to correct this pattern.
        genuine_re=re.compile(r"^original_(?P<writer>\d+)_\d+\.png$", re.IGNORECASE),
        forgery_re=re.compile(r"^forgeries_(?P<writer>\d+)_\d+\.png$", re.IGNORECASE),
    ),
    FlatFoldersSpec(
        name="Institutional (signature_verification)",
        genuine_dir=DATA_RAW / "signature_verification" / "full_org",
        forgery_dir=DATA_RAW / "signature_verification" / "full_forg",
        genuine_re=re.compile(r"^original_(?P<writer>\d+)_\d+\.jpg$", re.IGNORECASE),
        forgery_re=re.compile(r"^forgeries_(?P<writer>\d+)_\d+\.jpg$", re.IGNORECASE),
    ),
]


def scan_per_writer(spec: PerWriterFolderSpec) -> dict[int, WriterStats] | None:
    if not spec.root.is_dir():
        print(f"  [SKIP] Root folder not found: {spec.root}")
        return None

    stats: dict[int, WriterStats] = {}
    for wdir in sorted(p for p in spec.root.iterdir() if p.is_dir()):
        m = spec.folder_id_re.match(wdir.name)
        if not m:
            print(f"  [WARN] Folder name doesn't look like a writer ID: {wdir.name}")
            continue
        folder_id = int(m.group("writer"))
        ws = stats.setdefault(folder_id, WriterStats())

        for f in wdir.iterdir():
            if not f.is_file():
                continue
            fm = spec.forgery_re.match(f.name)
            gm = None if fm else spec.genuine_re.match(f.name)
            if fm:
                ws.forgery += 1
                file_writer = int(fm.group("writer"))
            elif gm:
                ws.genuine += 1
                file_writer = int(gm.group("writer"))
            else:
                if re.match(r"^ParamsUser\d+\.mat$", f.name, re.IGNORECASE):
                    continue
                ws.unrecognized_files.append(f.name)
                continue
            if file_writer != folder_id:
                ws.mismatched_files.append(
                    f"{f.name} (claims writer {file_writer}, in folder {folder_id})"
                )
    return stats


def scan_flat(spec: FlatFoldersSpec) -> dict[int, WriterStats] | None:
    missing = [d for d in (spec.genuine_dir, spec.forgery_dir) if not d.is_dir()]
    if missing:
        for d in missing:
            print(f"  [SKIP] Folder not found: {d}")
        return None

    stats: dict[int, WriterStats] = {}

    def orphan(name: str) -> WriterStats:
        # writer id -1 collects files whose pattern didn't match at all
        return stats.setdefault(-1, WriterStats())

    for f in spec.genuine_dir.iterdir():
        if not f.is_file():
            continue
        m = spec.genuine_re.match(f.name)
        if m:
            stats.setdefault(int(m.group("writer")), WriterStats()).genuine += 1
        else:
            orphan(f.name).unrecognized_files.append(f"full_org/{f.name}")

    for f in spec.forgery_dir.iterdir():
        if not f.is_file():
            continue
        m = spec.forgery_re.match(f.name)
        if m:
            stats.setdefault(int(m.group("writer")), WriterStats()).forgery += 1
        else:
            orphan(f.name).unrecognized_files.append(f"full_forg/{f.name}")

    return stats


def report(name: str, stats: dict[int, WriterStats]) -> None:
    real = {k: v for k, v in stats.items() if k != -1}
    orphans = stats.get(-1)

    genuine_counts = Counter(ws.genuine for ws in real.values())
    forgery_counts = Counter(ws.forgery for ws in real.values())
    modal_g = genuine_counts.most_common(1)[0][0] if genuine_counts else 0
    modal_f = forgery_counts.most_common(1)[0][0] if forgery_counts else 0

    print(f"  Writers found:        {len(real)}")
    print(f"  Total genuine files:  {sum(ws.genuine for ws in real.values())}")
    print(f"  Total forgery files:  {sum(ws.forgery for ws in real.values())}")
    print(f"  Modal counts/writer:  {modal_g} genuine, {modal_f} forgery")

    anomalies = 0
    for wid in sorted(real):
        ws = real[wid]
        problems = []
        if ws.genuine != modal_g:
            problems.append(f"genuine={ws.genuine} (modal {modal_g})")
        if ws.forgery != modal_f:
            problems.append(f"forgery={ws.forgery} (modal {modal_f})")
        if ws.mismatched_files:
            problems.append(f"{len(ws.mismatched_files)} mismatched writer ID")
        if ws.unrecognized_files:
            problems.append(f"{len(ws.unrecognized_files)} unrecognized")
        if problems:
            anomalies += 1
            print(f"    [ANOMALY] writer {wid}: " + "; ".join(problems))
            for d in ws.mismatched_files[:10]:
                print(f"       mismatch -> {d}")
            for d in ws.unrecognized_files[:10]:
                print(f"       unrecognized -> {d}")

    if orphans and orphans.unrecognized_files:
        anomalies += 1
        n = len(orphans.unrecognized_files)
        print(f"    [ANOMALY] {n} file(s) matching no pattern:")
        for d in orphans.unrecognized_files[:10]:
            print(f"       -> {d}")
        if n > 10:
            print(f"       ... and {n - 10} more (see CSV)")

    print("  Result: CLEAN" if anomalies == 0
          else f"  Result: {anomalies} anomaly group(s) — see above")


def write_csv(name: str, stats: dict[int, WriterStats], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"\W+", "_", name.lower()).strip("_")
    out = out_dir / f"counts_{safe}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["writer_id", "genuine", "forgery",
                    "mismatched_files", "unrecognized_files"])
        for wid in sorted(stats):
            ws = stats[wid]
            w.writerow([wid, ws.genuine, ws.forgery,
                        " | ".join(ws.mismatched_files),
                        " | ".join(ws.unrecognized_files)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="store_true",
                    help="write per-writer counts to report/dataset_checks/")
    args = ap.parse_args()

    csv_dir = PROJECT_ROOT / "report" / "dataset_checks"
    exit_code = 0

    for spec in DATASETS:
        print(f"\n=== {spec.name} ===")
        if isinstance(spec, PerWriterFolderSpec):
            stats = scan_per_writer(spec)
        else:
            stats = scan_flat(spec)
        if stats is None:
            exit_code = 1
            continue
        report(spec.name, stats)
        if args.csv:
            path = write_csv(spec.name, stats, csv_dir)
            print(f"  CSV written: {path.relative_to(PROJECT_ROOT)}")

    print()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())