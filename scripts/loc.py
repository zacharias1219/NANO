"""A9: count physical lines in nano/ and fail if the total reaches the 1500-line budget."""
import sys
from pathlib import Path

LIMIT = 1500
root = Path(__file__).resolve().parents[1] / "nano"
rows = sorted((len(p.read_text().splitlines()), p.name) for p in root.glob("*.py"))
for n, name in rows:
    print(f"{n:6d}  nano/{name}")
total = sum(n for n, _ in rows)
print(f"{total:6d}  total (limit {LIMIT})")
sys.exit(0 if total < LIMIT else 1)
