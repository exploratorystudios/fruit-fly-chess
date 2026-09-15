"""The fly-connectome chess model, vendored so this repo runs it standalone.

`brain.py` and `encoding.py` are copies of the trained model's own source; the
only change is an optional spike raster in the forward pass, so the visualiser can
show which neurons actually fired rather than inferring it.
"""
