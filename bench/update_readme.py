"""Copy results/bench.md into README.md between the BENCH markers."""
from pathlib import Path

readme, table = Path("README.md"), Path("results/bench.md").read_text(encoding="utf-8").strip()
text = readme.read_text(encoding="utf-8")
a, b = "<!-- BENCH:START -->", "<!-- BENCH:END -->"
start, end = text.index(a) + len(a), text.index(b)
readme.write_text(text[:start] + "\n" + table + "\n" + text[end:], encoding="utf-8")
print("README updated")
