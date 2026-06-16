"""egglog-hls: DialEgg/egglog track for round-tripping HLS designs through an e-graph.

Public surface is the round-trip pipeline in `roundtrip`.
"""

from .roundtrip import lower, raise_, round_trip

__all__ = ["lower", "raise_", "round_trip"]
