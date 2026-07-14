"""Space definitions for gymlx environments."""
import mlx.core as mx


class Box:
    """Continuous space with bounds.

    Parameters:
        low: Lower bound of the space (scalar or array).
        high: Upper bound of the space (scalar or array).
        shape: Shape of samples from the space.
        dtype: Data type of samples (default: mx.float32).
    """

    def __init__(
        self,
        low: mx.array,
        high: mx.array,
        shape: tuple,
        dtype: mx.Dtype = mx.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, key: mx.array = None) -> mx.array:
        """Generate a random sample from the space.

        Args:
            key: Optional MLX random key. If None, uses global RNG state.

        Returns:
            Random sample uniformly distributed between low and high.
        """
        return mx.random.uniform(
            low=self.low,
            high=self.high,
            shape=self.shape,
            dtype=self.dtype,
            key=key,
        )

    def __repr__(self) -> str:
        return f"Box(low={self.low}, high={self.high}, shape={self.shape}, dtype={self.dtype})"


class Discrete:
    """Discrete space with n actions (0 to n-1).

    Parameters:
        n: Number of discrete actions.
    """

    def __init__(self, n: int):
        self.n = n

    def sample(self, key: mx.array = None) -> mx.array:
        """Generate a random sample from the space.

        Args:
            key: Optional MLX random key. If None, uses global RNG state.

        Returns:
            Random integer in [0, n).
        """
        return mx.random.randint(low=0, high=self.n, shape=(), key=key)

    def __repr__(self) -> str:
        return f"Discrete(n={self.n})"
