"""Run the full project pipeline: dataset -> detection -> EDA (charts, Excel, PDF report)."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

for step in ["generate_dataset.py", "detection.py", "eda.py"]:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, step], cwd=SRC, check=True)

print("\nDone. See data/, figures/, outputs/ and reports/.")
