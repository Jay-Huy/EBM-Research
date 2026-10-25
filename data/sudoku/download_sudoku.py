"""Phase C.2 Task 5.1: download Sudoku-RRN and log the raw csv files.

Run from the repo root:
    python data/sudoku/download_sudoku.py

Logs (tests/logs/download_sudoku.log), for each of train.csv, valid.csv, test.csv:
    - path
    - number of columns and column names (or "no header row")
    - number of rows, shape
    - min / max number of given digits per board
    - one sample (board and solution) as a 9x9 grid
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.sudoku.sudoku_dataset import CSV_DIR, CSV_FILES, URL, SudokuRRN
from utils.logger import REPO_ROOT, TestLogger


def grid(s):
    """Return an 81-character board string as 9 lines, with the 3x3 boxes separated."""
    lines = []
    for r in range(9):
        row = s[9 * r: 9 * r + 9]
        lines.append(" ".join(row[0:3]) + " | " + " ".join(row[3:6]) + " | " + " ".join(row[6:9]))
        if r in (2, 5):
            lines.append("------+-------+------")
    return lines


def main():
    log = TestLogger("download_sudoku", config={"url": URL, "csv_dir": str(CSV_DIR.relative_to(REPO_ROOT))})
    fails = 0
    SudokuRRN.download(log=log.info)
    for name in CSV_FILES:
        path = CSV_DIR / name
        log.info(f"[{name}] path: {path.relative_to(REPO_ROOT)}")
        if not path.exists():
            log.error("    FAIL file not found")
            fails += 1
            continue
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        first = rows[0]
        if all(c.isdigit() for c in first):
            log.info(f"    columns: {len(first)}, no header row")
        else:
            log.info(f"    columns: {len(first)}, names: {first}")
            rows = rows[1:]
        log.info(f"    rows: {len(rows)}, shape: ({len(rows)}, {len(first)})")
        givens = [sum(c != "0" for c in r[0]) for r in rows]
        log.info(f"    given digits per board: min {min(givens)}, max {max(givens)}")
        log.info(f"    sample (row 0, {givens[0]} givens): column 0 (0 = empty) | column 1")
        for b, s in zip(grid(rows[0][0]), grid(rows[0][1])):
            log.info(f"        {b}      {s}")
    (log.info if fails == 0 else log.error)(f"RESULT: {'PASS' if fails == 0 else f'FAIL ({fails} errors)'}")
    log.close()
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
