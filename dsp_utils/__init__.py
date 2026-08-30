# 1. Pull everything into the top-level namespace
from .processing import *
from .SignalPlotter import *
from .AudioFeatureExtractor import *

# 2. Correctly import the modules to access their attributes
from . import processing
from . import SignalPlotter
from . import AudioFeatureExtractor

# 3. Combine the __all__ lists
# We use getattr() as a safety measure in case a file is missing __all__
__all__ = getattr(processing, "__all__", []) + getattr(SignalPlotter, "__all__", [])