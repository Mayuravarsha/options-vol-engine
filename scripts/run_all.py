#!/usr/bin/env python3
"""Run the validation suite and every demo script, in order.

    python scripts/run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

STEPS = [
    ("validation suite", [sys.executable, os.path.join(ROOT, "tests", "test_all.py")]),
    ("1. pricing, Greeks, implied vol",
     [sys.executable, os.path.join(HERE, "01_pricing_greeks_iv.py")]),
    ("2. volatility surface",
     [sys.executable, os.path.join(HERE, "02_volatility_surface.py")]),
    ("3. Monte Carlo",
     [sys.executable, os.path.join(HERE, "03_monte_carlo.py")]),
    ("4. GARCH forecasting",
     [sys.executable, os.path.join(HERE, "04_garch_forecasting.py")]),
]


def main() -> None:
    results = []
    for name, cmd in STEPS:
        print("\n" + "#" * 84)
        print(f"#  {name}")
        print("#" * 84)
        t0 = time.perf_counter()
        rc = subprocess.call(cmd, cwd=ROOT)
        dt = time.perf_counter() - t0
        results.append((name, rc, dt))

    print("\n" + "=" * 84)
    print("  SUMMARY")
    print("=" * 84)
    for name, rc, dt in results:
        print(f"    {'OK  ' if rc == 0 else 'FAIL'}  {name:<42}{dt:>8.1f}s")
    print(f"\n  charts and JSON written to {os.path.abspath(os.path.join(ROOT, 'results'))}\n")
    sys.exit(0 if all(rc == 0 for _, rc, _ in results) else 1)


if __name__ == "__main__":
    main()
