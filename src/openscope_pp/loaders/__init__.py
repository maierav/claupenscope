"""Per-technique NWB loaders that produce a common trial representation."""

from openscope_pp.loaders.streaming import open_nwb
from openscope_pp.loaders.trials import load_trials
from openscope_pp.loaders.responses import load_responses
from openscope_pp.loaders.behavior import load_behavior

__all__ = ["open_nwb", "load_trials", "load_responses", "load_behavior"]
