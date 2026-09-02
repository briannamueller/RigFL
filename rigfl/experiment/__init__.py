"""Experiment layer: turn a config into a run, and runs into a table.

This is the thin CLI/orchestration around the pure ``core`` engine:
  * ``device``   -- resolve cpu / mps / cuda / auto.
  * ``registry`` -- build any algorithm by name with validated configuration.
  * ``run``      -- one (algorithm, seed) run: load partition -> train -> save JSON.
  * ``collect``  -- many JSONs -> multi-seed mean±std table.

The split keeps the engine free of argparse/IO; everything here is replaceable
without touching an algorithm or the loop.
"""
