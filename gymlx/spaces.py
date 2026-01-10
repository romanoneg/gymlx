"""Space definitions for gymlx environments."""
from typing import NamedTuple
import mlx.core as mx


class Box(NamedTuple):
    """Continuous space with bounds."""
    low: mx.array
    high: mx.array
    shape: tuple
    dtype: mx.Dtype = mx.float32


class Discrete(NamedTuple):
    """Discrete space with n actions (0 to n-1)."""
    n: int
