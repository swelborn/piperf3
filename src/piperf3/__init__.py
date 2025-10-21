"""Python wrapper for iperf3 with configuration management and plotting."""

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

__version__ = "0.0.9"

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
    "main",
]


def main() -> None:
    """Main entry point for the CLI."""
    app()
