"""Fit-once, score-many pipeline of the triage prototype (models_plan.md §29).

`python -m src.pipeline fit` checks and lists the frozen artifacts that the
modeling stages produced; `python -m src.pipeline score` takes a new table
with the CFPB columns and returns, without retraining anything, the five
signals per complaint and the counts, predictive intervals and alerts per
week and per day.
"""
