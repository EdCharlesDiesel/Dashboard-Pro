"""15M Fibonacci Entry — Bloomberg-terminal entry point.

Implementation lives in src/pages_lib/fib_entry.py. Merges the former
15M Rejection and 15M Entry Signal pages into one Fibonacci-retracement trigger
driven by Setup Ranker signals.
"""
from src.pages_lib.fib_entry import FibEntryPage

FibEntryPage().run()
