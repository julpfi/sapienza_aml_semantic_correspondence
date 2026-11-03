"""TODO: run a model on the validation split, save per-query predictions, report PCK.

metric.py is not in the competition download, so implement PCK@0.10 from the Evaluation
page: distance to the nearest accepted target, divided by the object size L. train.csv has
a `scale` column that looks like L. Confirm this before trusting local scores.
"""
