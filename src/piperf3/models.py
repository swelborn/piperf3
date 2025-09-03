import json
import os
import socket
from datetime import datetime, timezone
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typer import Option


class BuilderBase(BaseSettings):
    """Mixin to build CLI arguments from Typer Option fields in a Pydantic model."""

    def build_cli_args(self) -> list[str]:
        args: list[str] = []
        for field_name, field_info in self.__class__.model_fields.items():
            value = getattr(self, field_name)
            if value is None or not isinstance(
                field_info.default, typer.models.OptionInfo
            ):
                continue

            opt_name = self._get_option_name(field_name, field_info.default)
            self._append_value(args, opt_name, value)
        return args

    @staticmethod
    def _get_option_name(field_name: str, opt_info: typer.models.OptionInfo) -> str:
        if opt_info.param_decls:
            option_names = sorted(
                opt_info.param_decls, key=lambda x: (not x.startswith("--"), x)
            )
            return option_names[0]
        return f"--{field_name.replace('_', '-')}"

    @staticmethod
    def _append_value(args: list[str], opt_name: str, value: Any) -> None:
        if isinstance(value, bool):
            if value:
                args.append(opt_name)
            return
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, list):
            for v in value:
                args.extend([opt_name, str(v)])
            return
        args.extend([opt_name, str(value)])


class IperfMode(StrEnum):
    CLIENT = "client"
    SERVER = "server"


class Protocol(StrEnum):
    TCP = "tcp"
    UDP = "udp"
    SCTP = "sctp"


class Format(StrEnum):
    KBITS = "k"
    MBITS = "m"
    GBITS = "g"
    TBITS = "t"
    KBYTES = "K"
    MBYTES = "M"
    GBYTES = "G"
    TBYTES = "T"


class CongestionAlgorithm(StrEnum):
    CUBIC = "cubic"
    RENO = "reno"
    BBR = "bbr"
    VEGAS = "vegas"
    WESTWOOD = "westwood"
    BIC = "bic"
    HTCP = "htcp"


class IperfBaseConfig(BuilderBase):
    """Base configuration with common iperf3 options."""

    mode: IperfMode = Field(frozen=True)

    # Connection settings
    port: int | None = Option(
        5201,
        "-p",
        "--port",
        parser=int,
        min=1,
        max=65535,
        help="Server port to listen on/connect to (default 5201)",
    )
    bind_address: str | None = Option(
        None,
        "--bind",
        "-B",
        help="Bind to the specific interface/address (host[%dev] for IPv6 link-local)",
    )
    bind_device: str | None = Option(
        None,
        "--bind-dev",
        help="Bind to the specified network interface (SO_BINDTODEVICE, may require root)",
    )

    # Protocol
    ipv4_only: bool = Option(False, "-4", "--version4", help="Only use IPv4")
    ipv6_only: bool = Option(False, "-6", "--version6", help="Only use IPv6")

    # Output
    format: Format | None = Option(
        None,
        "-f",
        "--format",
        parser=str,
        help="Format to report: k/m/g/t for bits, K/M/G/T for bytes",
    )
    interval: float | None = Option(
        None,
        "-i",
        "--interval",
        parser=int,
        min=0,
        help="Pause n seconds between periodic throughput reports (default 1, 0 to disable)",
    )
    json_output: bool = Option(
        False, "-J", "--json", help="Output results in JSON format"
    )
    json_stream: bool = Option(
        False,
        "--json-stream",
        help="Output in line-delimited JSON format (real-time parsable)",
    )
    verbose: bool = Option(False, "-V", "--verbose", help="Give more detailed output")

    # Advanced
    cpu_affinity: str | None = Option(
        None,
        "-A",
        "--affinity",
        parser=str,
        help="Set CPU affinity: 'n' for local, 'n,m' to override server's affinity",
    )
    pidfile: Path | None = Option(
        None, "-I", "--pidfile", parser=str, help="Write a file with the process ID"
    )
    logfile: Path | None = Option(
        None, "--logfile", parser=str, help="Send output to a log file"
    )
    force_flush: bool = Option(
        False, "--forceflush", help="Force flushing output at every interval"
    )
    timestamps: str | None = Option(
        None,
        "--timestamps",
        help="Prepend a timestamp to each output line (optional strftime format)",
    )
    debug: bool = Option(
        False, "-d", "--debug", help="Emit debugging output (developer use)"
    )
    dscp: str | None = Option(None, "--dscp", help="set the IP DSCP bits")

    # Timeouts
    rcv_timeout: int | None = Option(
        None,
        "--rcv-timeout",
        parser=int,
        min=0,
        help="Idle timeout for receiving data during active tests (ms, default 120000)",
    )
    snd_timeout: int | None = Option(
        None,
        "--snd-timeout",
        parser=int,
        min=0,
        help="Timeout for unacknowledged TCP data (ms)",
    )

    # Security
    use_pkcs1_padding: bool = Option(
        False,
        "--use-pkcs1-padding",
        help="Use less secure PKCS1 padding for RSA authentication (compatibility mode)",
    )

    # MPTCP
    mptcp: bool = Option(False, "-m", "--mptcp", help="Enable MPTCP usage (TCP only)")

    @field_validator("cpu_affinity")
    @classmethod
    def validate_cpu_affinity(cls, v):
        if v is None:
            return v
        if isinstance(v, int):
            return str(v)
        if isinstance(v, str):
            parts = v.split(",")
            if len(parts) > 2:
                raise ValueError("CPU affinity must be 'n' or 'n,m'")
            for part in parts:
                int(part.strip())
        return v

    @model_validator(mode="after")
    def validate_ip_versions(self):
        if self.ipv4_only and self.ipv6_only:
            raise ValueError("Cannot specify both IPv4-only and IPv6-only")
        return self


class IperfServerConfig(IperfBaseConfig):
    model_config = SettingsConfigDict(
        env_prefix="IPERF_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )
    """Server-specific configuration."""

    mode: IperfMode = Field(default=IperfMode.SERVER, frozen=True)

    daemon: bool = Option(
        False, "-D", "--daemon", help="Run the server in background as a daemon"
    )
    one_off: bool = Option(
        False, "-1", "--one-off", help="Handle one client connection, then exit"
    )
    idle_timeout: int | None = Option(
        None,
        "--idle-timeout",
        parser=str,
        min=0,
        help="Restart or exit after n seconds of idle time",
    )
    server_bitrate_limit: str | None = Option(
        None,
        "--server-bitrate-limit",
        help="Abort if client requests >n bits/sec or exceeds average rate",
    )
    rsa_private_key_path: Path | None = Option(
        None,
        "--rsa-private-key-path",
        help="Path to RSA private key for decrypting authentication credentials",
    )
    authorized_users_path: Path | None = Option(
        None,
        "--authorized-users-path",
        help="Path to file containing authorized user credentials",
    )
    time_skew_threshold: int | None = Option(
        None,
        "--time-skew-threshold",
        parser=str,
        min=0,
        help="Time skew threshold (seconds) in authentication",
    )


class IperfClientConfig(IperfBaseConfig):
    """Client-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="IPERF_CLIENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    mode: IperfMode = Field(default=IperfMode.CLIENT, frozen=True)

    server_host: str | None = Field(default=None, description="Server hostname or IP")

    time: int | None = Option(
        None,
        "-t",
        "--time",
        parser=int,
        min=1,
        help="Time in seconds to transmit for (default 10)",
    )
    bytes: str | None = Option(
        None,
        "-n",
        "--bytes",
        help="Number of bytes to transmit instead of time-based test",
    )
    blockcount: str | None = Option(
        None,
        "-k",
        "--blockcount",
        help="Number of blocks (packets) to transmit instead of time/bytes",
    )
    bitrate: str | None = Option(
        None,
        "-b",
        "--bitrate",
        help="Target bitrate (bits/sec), supports suffixes and burst mode",
    )
    pacing_timer: str | None = Option(
        None,
        "--pacing-timer",
        parser=str,
        min=1,
        help="Pacing timer interval in microseconds (default 1000)",
    )
    fq_rate: str | None = Option(
        None, "--fq-rate", help="Fair-queueing based socket pacing rate (bits/sec)"
    )
    connect_timeout: int | None = Option(
        None,
        "--connect-timeout",
        parser=int,
        min=1,
        help="Timeout for establishing control connection (ms)",
    )
    parallel_streams: int | None = Option(
        None,
        "-P",
        "--parallel",
        parser=int,
        min=1,
        help="Number of parallel client streams",
    )
    reverse: bool = Option(
        False, "-R", "--reverse", help="Reverse direction: server sends to client"
    )
    bidir: bool = Option(
        False, "--bidir", help="Test in both directions simultaneously"
    )
    length: str | None = Option(
        None, "-l", "--length", help="Buffer length to read/write (TCP default 128KB)"
    )
    window: str | None = Option(
        None, "-w", "--window", help="Socket buffer/window size"
    )
    set_mss: int | None = Option(
        None,
        "-M",
        "--set-mss",
        parser=int,
        min=1,
        help="Set TCP/SCTP maximum segment size",
    )
    no_delay: bool = Option(
        False, "-N", "--no-delay", help="Disable Nagle's Algorithm (TCP/SCTP)"
    )
    client_port: int | None = Option(
        None,
        "--cport",
        parser=int,
        min=1,
        max=65535,
        help="Bind data streams to a specific client port",
    )
    tos: str | None = Option(
        None,
        "-S",
        "--tos",
        parser=str,
        help="Set IP type of service (TOS)",
    )
    flowlabel: int | None = Option(
        None, "-L", "--flowlabel", parser=str, min=0, help="Set IPv6 flow label"
    )
    file_input: Path | None = Option(
        None, "-F", "--file", help="Use a file as source/sink instead of generated data"
    )
    sctp_streams: int | None = Option(
        None, "--nstreams", parser=int, min=1, help="Number of SCTP streams"
    )
    zerocopy: bool = Option(
        False, "-Z", "--zerocopy", help="Use zero-copy method for sending data"
    )
    skip_rx_copy: bool = Option(
        False, "--skip-rx-copy", help="Ignore received packet data (MSG_TRUNC)"
    )
    omit: int | None = Option(
        None,
        "-O",
        "--omit",
        parser=int,
        min=0,
        help="Omit first N seconds from statistics (skip slow-start)",
    )
    title: str | None = Option(
        None, "-T", "--title", help="Prefix every output line with this string"
    )
    extra_data: str | None = Option(
        None, "--extra-data", help="Extra data string to include in JSON output"
    )
    congestion_algorithm: CongestionAlgorithm | None = Option(
        None, "-C", "--congestion", help="Set congestion control algorithm"
    )
    get_server_output: bool = Option(
        False, "--get-server-output", help="Retrieve server-side output"
    )
    udp_counters_64bit: bool = Option(
        False, "--udp-counters-64bit", help="Use 64-bit counters in UDP test packets"
    )
    repeating_payload: bool = Option(
        False,
        "--repeating-payload",
        help="Use repeating pattern in payload instead of random bytes",
    )
    dont_fragment: bool = Option(
        False, "--dont-fragment", help="Set IPv4 Don't Fragment bit (UDP over IPv4)"
    )
    username: str | None = Option(
        None, "--username", help="Username for authentication (if built with OpenSSL)"
    )
    rsa_public_key_path: Path | None = Option(
        None,
        "--rsa-public-key-path",
        help="Path to RSA public key for encrypting authentication credentials",
    )


class SlurmConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SLURM_", case_sensitive=False, extra="ignore"
    )

    job_id: str | None = None
    job_name: str | None = None
    job_partition: str | None = None
    job_account: str | None = None
    job_num_nodes: str | None = None
    ntasks: str | None = None
    cpus_per_task: str | None = None
    mem_per_node: str | None = None
    nodelist: str | None = None
    procid: str | None = None
    localid: str | None = None


class GeneralConfig(BuilderBase):
    model_config = SettingsConfigDict(
        env_prefix="PIPERF3_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # CLI-exposed options
    name: str = Option(
        "iperf3_test_environment", "--name", help="Name of the test environment"
    )
    description: str | None = Option(
        None, "--description", help="Description of the test environment"
    )
    version: str = Option("1.0", "--version", help="Version of the configuration")
    created_by: str | None = Option(
        os.environ.get("USER", "unknown"),
        "--created-by",
        help="User who created this configuration",
    )
    tags: list[str] = Option(
        [], "--tag", help="One or more tags for this configuration"
    )
    output_directory: Path | None = Option(
        None,
        "--output-directory",
        "--output-dir",
        help="Directory to store output files",
    )
    run_id: str | None = Option(None, "--run-id", help="Run identifier")

    # Internal / auto-generated fields (not exposed as CLI options)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_info: dict[str, Any] = Field(default_factory=lambda: _collect_node_info())

    def get_provenance(self) -> dict[str, Any]:
        slurm_config = SlurmConfig()
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
            "node_info": self.node_info,
            "slurm_info": slurm_config.model_dump(),
        }


class IntervalStream(BaseModel):
    socket: int
    start: float
    end: float
    seconds: float
    bytes: int
    bits_per_second: float
    omitted: bool
    sender: bool


class IntervalSum(BaseModel):
    start: float
    end: float
    seconds: float
    bytes: int
    bits_per_second: float
    omitted: bool
    sender: bool


class Interval(BaseModel):
    streams: list[IntervalStream]
    sum: IntervalSum


class SumSentReceived(BaseModel):
    start: float
    end: float
    seconds: float
    bytes: int
    bits_per_second: float
    sender: bool


class CpuUtilization(BaseModel):
    host_total: float
    host_user: float
    host_system: float
    remote_total: float
    remote_user: float
    remote_system: float


class EndSection(BaseModel):
    streams: list[dict]  # Could be typed more strictly
    sum_sent: SumSentReceived
    sum_received: SumSentReceived
    cpu_utilization_percent: CpuUtilization


class StartTestStart(BaseModel):
    protocol: str
    num_streams: int
    blksize: int
    omit: int
    duration: int
    bytes: int
    blocks: int
    reverse: int
    tos: int
    target_bitrate: int
    bidir: int
    fqrate: int
    interval: int


class StartSection(BaseModel):
    connected: list[dict]
    version: str
    system_info: str
    timestamp: dict
    connecting_to: dict
    cookie: str
    tcp_mss_default: int
    target_bitrate: int
    fq_rate: int
    sock_bufsize: int
    sndbuf_actual: int
    rcvbuf_actual: int
    test_start: StartTestStart


class Iperf3JsonResult(BaseModel):
    start: StartSection
    intervals: list[Interval]
    end: EndSection


def _read_proc_value(path: str, key: str, transform=lambda v: v) -> Any:
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith(key):
                    return transform(
                        line.split(":", 1)[1].strip()
                        if ":" in line
                        else line.split()[1]
                    )
    except Exception:
        return None


def _collect_node_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "cpu_model": _read_proc_value("/proc/cpuinfo", "model name"),
        "cpu_count": str(
            _read_proc_value(
                "/proc/cpuinfo",
                "processor",
                lambda _: open("/proc/cpuinfo").read().count("processor"),
            )
        ),
        "memory_gb": _read_proc_value(
            "/proc/meminfo",
            "MemTotal",
            lambda v: str(round(int(v.split()[0]) / 1024 / 1024, 2)),
        ),
    }


class IperfResult(BaseModel):
    environment_name: str
    run_id: str
    start_time: datetime
    end_time: datetime | None = None
    config_used: IperfClientConfig | IperfServerConfig
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    json_results: Iperf3JsonResult | None = None
    output_directory: Path
    stdout_file: Path | None = None
    stderr_file: Path | None = None
    json_file: Path | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    def save_to_directory(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._write_if_content(directory / "stdout.txt", self.stdout)
        self._write_if_content(directory / "stderr.txt", self.stderr)
        if self.json_results:
            self._write_json(directory / "results.json", self.json_results.model_dump())
        self._write_json(
            directory / "result_metadata.json",
            self.model_dump(exclude={"stdout", "stderr", "json_results"}),
        )

    @staticmethod
    def _write_if_content(path: Path, content: str) -> None:
        if content:
            path.write_text(content)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
