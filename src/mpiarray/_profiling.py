"""The tag of mpiarray's scope-profiler regions, which are off unless enabled.

``mpiarray/__init__`` registers ``"mpiarray"`` as an opt-in tag, so the
regions record nothing unless the application enables the tag, e.g. with
``scope_profiler.setup(enable_tags=["mpiarray"])`` or
``SCOPE_PROFILER_ENABLE_TAGS=mpiarray``. A disabled region costs well under a
microsecond per call, so only coarse operations (communication, I/O, whole
array kernels) carry one.
"""

#: The tags of every mpiarray region; region names start with ``"mpiarray."``.
PROFILE_TAGS = ("mpiarray",)
