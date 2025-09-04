import importlib.util

if importlib.util.find_spec("piperf3") is None:
    raise RuntimeError("Could not import piperf3")
