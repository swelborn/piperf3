"""Python wrapper for iperf3 with configuration management and plotting."""

from ._version import __version__
from .cli import app

# from .config import create_example_configs  # Temporarily commented out
from .models import (
    CongestionAlgorithm,
    Format,
    GeneralConfig,
    IperfBaseConfig,
    IperfClientConfig,
    IperfMode,
    IperfResult,
    IperfServerConfig,
    Protocol,
    SlurmConfig,
)
from .plotting import IperfPlotter
from .runner import Iperf3Runner

__all__ = [
    "CongestionAlgorithm",
    "Format",
    "GeneralConfig",
    "Iperf3Runner",
    "IperfBaseConfig",
    "IperfClientConfig",
    "IperfMode",
    "IperfPlotter",
    "IperfResult",
    "IperfServerConfig",
    "Protocol",
    "SlurmConfig",
    "__version__",
    "main",
]


def main() -> None:
    """Main entry point for the CLI."""
    app()
