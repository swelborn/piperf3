import getpass
import json
import os
import platform
import socket
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from piperf3._version import __version__
from piperf3.constants import (
    JSON_RESULTS_FILENAME,
    METADATA_FILENAME,
    STDERR_FILENAME,
    STDOUT_FILENAME,
)

T = TypeVar("T")


class TyperOptionSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    default: Any = None
    param_decls: list[str] = []
    help: str | None = None
    min: int | None = None
    max: int | None = None
    parser: Callable | None = None


def CLIField(default: T = None, *param_decls, **kwargs) -> T:
    """
    Helper to create a Pydantic field with Typer Option metadata stored in json_schema_extra['cli'].
    This allows Pydantic to use normal defaults for env loading while still keeping CLI info.
    """
    option = TyperOptionSchema(default=default, param_decls=list(param_decls), **kwargs)
    return Field(
        default=default,
        json_schema_extra={"cli": option.model_dump()},
    )


class BuilderBase(BaseSettings):
    """Mixin to build iperf CLI arguments from Typer Option fields in a Pydantic model."""

    def build_cli_args(self) -> list[str]:
        args: list[str] = []
        for field_name, field_info in self.__class__.model_fields.items():
            value = getattr(self, field_name)
            meta = (
                field_info.json_schema_extra.get("cli")
                if field_info.json_schema_extra
                else None
            )
            if value is None or not meta or not isinstance(meta, dict):
                continue
            meta = TyperOptionSchema(**meta)  # type: ignore
            opt_name = self._get_option_name(field_name, meta)
            self._append_value(args, opt_name, value)
        return args

    @staticmethod
    def _get_option_name(field_name: str, cli_meta: TyperOptionSchema) -> str:
        param_decls = cli_meta.param_decls
        if param_decls:
            option_names = sorted(
                param_decls, key=lambda x: (not x.startswith("--"), x)
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

    def pretty_print(self):
        """Format configuration object showing all non-None fields."""
        lines = []
        for field_name, field_info in self.__class__.model_fields.items():
            value = getattr(self, field_name)
            if value is not None and value != field_info.default:
                display_name = field_name.replace("_", " ").upper()
                lines.append(f"{display_name}: {value}")

        return "\n".join(lines) if lines else "No configuration values set"


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

    # Connection settings
    port: int | None = CLIField(
        5201,
        "-p",
        "--port",
        parser=int,
        min=1,
        max=65535,
        help="Server port to listen on/connect to (default 5201)",
    )
    bind_address: str | None = CLIField(
        None, "--bind", "-B", help="Bind to the specific interface/address"
    )
    bind_device: str | None = CLIField(
        None, "--bind-dev", help="Bind to the specified network interface"
    )

    # Protocol
    ipv4_only: bool = CLIField(False, "-4", "--version4", help="Only use IPv4")
    ipv6_only: bool = CLIField(False, "-6", "--version6", help="Only use IPv6")

    # Output
    format: Format | None = CLIField(
        None,
        "-f",
        "--format",
        parser=str,
        help="Format to report: k/m/g/t for bits, K/M/G/T for bytes",
    )
    interval: float | None = CLIField(
        None,
        "-i",
        "--interval",
        parser=int,
        min=0,
        help="Pause n seconds between periodic throughput reports",
    )
    json_output: bool = CLIField(
        False, "-J", "--json", help="Output results in JSON format"
    )
    json_stream: bool = CLIField(
        False, "--json-stream", help="Output in line-delimited JSON format"
    )
    verbose: bool = CLIField(False, "-V", "--verbose", help="Give more detailed output")

    # Advanced
    cpu_affinity: str | None = CLIField(
        None, "-A", "--affinity", parser=str, help="Set CPU affinity"
    )
    pidfile: Path | None = CLIField(
        None, "-I", "--pidfile", parser=str, help="Write a file with the process ID"
    )
    logfile: Path | None = CLIField(
        None, "--logfile", parser=str, help="Send output to a log file"
    )
    force_flush: bool = CLIField(
        False, "--forceflush", help="Force flushing output at every interval"
    )
    timestamps: str | None = CLIField(
        None, "--timestamps", help="Prepend a timestamp to each output line"
    )
    debug: bool = CLIField(False, "-d", "--debug", help="Emit debugging output")
    dscp: str | None = CLIField(None, "--dscp", help="Set the IP DSCP bits")

    # Timeouts
    rcv_timeout: int | None = CLIField(
        None, "--rcv-timeout", parser=int, min=0, help="Idle timeout for receiving data"
    )
    snd_timeout: int | None = CLIField(
        None,
        "--snd-timeout",
        parser=int,
        min=0,
        help="Timeout for unacknowledged TCP data",
    )

    # Security
    use_pkcs1_padding: bool = CLIField(
        False, "--use-pkcs1-padding", help="Use less secure PKCS1 padding"
    )

    # MPTCP
    mptcp: bool = CLIField(False, "-m", "--mptcp", help="Enable MPTCP usage")

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

    mode: IperfMode = Field(default=IperfMode.SERVER, frozen=True)

    daemon: bool = CLIField(
        False, "-D", "--daemon", help="Run the server in background as a daemon"
    )
    one_off: bool = CLIField(
        False, "-1", "--one-off", help="Handle one client connection, then exit"
    )
    idle_timeout: int | None = CLIField(
        None,
        "--idle-timeout",
        parser=str,
        min=0,
        help="Restart or exit after n seconds of idle time",
    )
    server_bitrate_limit: str | None = CLIField(
        None, "--server-bitrate-limit", help="Abort if client requests >n bits/sec"
    )
    rsa_private_key_path: Path | None = CLIField(
        None, "--rsa-private-key-path", help="Path to RSA private key"
    )
    authorized_users_path: Path | None = CLIField(
        None, "--authorized-users-path", help="Path to file containing authorized users"
    )
    time_skew_threshold: int | None = CLIField(
        None,
        "--time-skew-threshold",
        parser=str,
        min=0,
        help="Time skew threshold in authentication",
    )


class IperfClientConfig(IperfBaseConfig):
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

    time: int | None = CLIField(
        None, "-t", "--time", parser=int, min=1, help="Time in seconds to transmit for"
    )
    bytes: str | None = CLIField(
        None, "-n", "--bytes", help="Number of bytes to transmit"
    )
    blockcount: str | None = CLIField(
        None, "-k", "--blockcount", help="Number of blocks to transmit"
    )
    bitrate: str | None = CLIField(None, "-b", "--bitrate", help="Target bitrate")
    pacing_timer: str | None = CLIField(
        None,
        "--pacing-timer",
        parser=str,
        min=1,
        help="Pacing timer interval in microseconds",
    )
    fq_rate: str | None = CLIField(
        None, "--fq-rate", help="Fair-queueing based socket pacing rate"
    )
    connect_timeout: int | None = CLIField(
        None,
        "--connect-timeout",
        parser=int,
        min=1,
        help="Timeout for establishing connection",
    )
    parallel_streams: int | None = CLIField(
        None,
        "-P",
        "--parallel",
        parser=int,
        min=1,
        help="Number of parallel client streams",
    )
    reverse: bool = CLIField(False, "-R", "--reverse", help="Reverse direction")
    bidir: bool = CLIField(
        False, "--bidir", help="Test in both directions simultaneously"
    )
    length: str | None = CLIField(
        None, "-l", "--length", help="Buffer length to read/write"
    )
    window: str | None = CLIField(
        None, "-w", "--window", help="Socket buffer/window size"
    )
    set_mss: int | None = CLIField(
        None,
        "-M",
        "--set-mss",
        parser=int,
        min=1,
        help="Set TCP/SCTP maximum segment size",
    )
    no_delay: bool = CLIField(
        False, "-N", "--no-delay", help="Disable Nagle's Algorithm"
    )
    client_port: int | None = CLIField(
        None,
        "--cport",
        parser=int,
        min=1,
        max=65535,
        help="Bind data streams to a specific client port",
    )
    tos: str | None = CLIField(
        None, "-S", "--tos", parser=str, help="Set IP type of service"
    )
    flowlabel: int | None = CLIField(
        None, "-L", "--flowlabel", parser=str, min=0, help="Set IPv6 flow label"
    )
    file_input: Path | None = CLIField(
        None, "-F", "--file", help="Use a file as source/sink"
    )
    sctp_streams: int | None = CLIField(
        None, "--nstreams", parser=int, min=1, help="Number of SCTP streams"
    )
    zerocopy: bool = CLIField(False, "-Z", "--zerocopy", help="Use zero-copy method")
    skip_rx_copy: bool = CLIField(
        False, "--skip-rx-copy", help="Ignore received packet data"
    )
    omit: int | None = CLIField(
        None,
        "-O",
        "--omit",
        parser=int,
        min=0,
        help="Omit first N seconds from statistics",
    )
    title: str | None = CLIField(
        None, "-T", "--title", help="Prefix every output line with this string"
    )
    extra_data: str | None = CLIField(
        None, "--extra-data", help="Extra data string for JSON output"
    )
    congestion_algorithm: CongestionAlgorithm | None = CLIField(
        None, "-C", "--congestion", help="Set congestion control algorithm"
    )
    get_server_output: bool = CLIField(
        False, "--get-server-output", help="Retrieve server-side output"
    )
    udp_counters_64bit: bool = CLIField(
        False, "--udp-counters-64bit", help="Use 64-bit counters in UDP test packets"
    )
    repeating_payload: bool = CLIField(
        False, "--repeating-payload", help="Use repeating pattern in payload"
    )
    dont_fragment: bool = CLIField(
        False, "--dont-fragment", help="Set IPv4 Don't Fragment bit"
    )
    username: str | None = CLIField(
        None, "--username", help="Username for authentication"
    )
    rsa_public_key_path: Path | None = CLIField(
        None, "--rsa-public-key-path", help="Path to RSA public key"
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
    mpi_type: str | None = None
    step_id: str | None = None
    nodeid: str | None = None
    task_pid: str | None = None
    prio_process: str | None = None
    submit_dir: str | None = None
    job_licenses: str | None = None
    srun_comm_host: str | None = None
    job_gid: str | None = None
    job_end_time: str | None = None
    tasks_per_node: str | None = None
    nnodes: str | None = None
    launch_node_ipaddr: str | None = None
    step_tasks_per_node: str | None = None
    job_start_time: str | None = None
    job_nodelist: str | None = None
    cluster_name: str | None = None
    job_cpus_per_node: str | None = None
    topology_addr: str | None = None
    step_nodelist: str | None = None
    srun_comm_port: str | None = None
    jobid: str | None = None
    job_qos: str | None = None
    topology_addr_pattern: str | None = None
    cpus_on_node: str | None = None
    job_uid: str | None = None
    script_context: str | None = None
    pty_win_row: str | None = None
    job_user: str | None = None
    pty_win_col: str | None = None
    stepmgr: str | None = None
    submit_host: str | None = None
    step_launcher_port: str | None = None
    pty_port: str | None = None
    gtids: str | None = None
    step_num_tasks: str | None = None
    step_num_nodes: str | None = None
    oom_kill_step: str | None = None


class NodeInfo(BaseModel):
    hostname: str = Field(default_factory=socket.gethostname)
    fqdn: str = Field(default_factory=socket.getfqdn)
    cpu_count: int | None = Field(default_factory=os.cpu_count)
    username: str = Field(default_factory=getpass.getuser)
    platform_system: str = Field(default_factory=platform.system)
    platform_release: str = Field(default_factory=platform.release)
    platform_version: str = Field(default_factory=platform.version)
    platform_machine: str = Field(default_factory=platform.machine)
    platform_processor: str = Field(default_factory=platform.processor)
    architecture: tuple[str, str] = Field(default_factory=platform.architecture)
    python_version: str = Field(default_factory=platform.python_version)
    python_build: tuple[str, str] = Field(default_factory=platform.python_build)
    python_compiler: str = Field(default_factory=platform.python_compiler)
    node: str = Field(default_factory=platform.node)
    piperf3_version: str = Field(default_factory=lambda: __version__)


class GeneralConfig(BuilderBase):
    model_config = SettingsConfigDict(
        env_prefix="PIPERF3_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )
    iperf3_path: Path = CLIField(
        Path("iperf3"),
        "--iperf3-path",
        help="Path to the iperf3 executable",
    )
    created_by: str | None = CLIField(
        os.environ.get("USER", "unknown"),
        "--created-by",
        help="User who created this configuration",
    )
    output_directory: Path = CLIField(
        Path("./results"),
        "--output-directory",
        "--output-dir",
        help="Directory to store output files",
    )
    run_id: str | None = CLIField(None, "--run-id", help="Run identifier")
    results_dir_timestamp: bool = CLIField(
        True,
        "--results-dir-timestamp",
        help="Whether to add a timestamp prefix to output directory",
    )
    slurm: SlurmConfig = Field(
        default_factory=SlurmConfig, description="SLURM job info"
    )
    numa_node: int | None = CLIField(
        None, "--numa-node", parser=int, min=0, help="NUMA node to bind to"
    )


class StreamBase(BaseModel):
    socket: int
    start: float
    end: float
    seconds: float
    bytes: int
    bits_per_second: float
    sender: bool
    retransmits: int | None = None
    snd_cwnd: int | None = None
    snd_wnd: int | None = None
    rtt: int | None = None
    rttvar: int | None = None
    pmtu: int | None = None
    reorder: int | None = None
    max_snd_cwnd: int | None = None
    max_snd_wnd: int | None = None
    max_rtt: int | None = None
    min_rtt: int | None = None
    mean_rtt: int | None = None

    @property
    def throughput_gbps(self) -> float:
        """Throughput in Gbps."""
        return self.bits_per_second / 1e9

    @property
    def data_mb(self) -> float:
        """Data transferred in MB."""
        return self.bytes / 1e6

    @property
    def time(self) -> float:
        """Alias for start time."""
        return self.start


class IntervalStream(StreamBase):
    omitted: bool
    retransmits: int | None = None
    snd_cwnd: int | None = None
    snd_wnd: int | None = None
    rtt: int | None = None
    rttvar: int | None = None
    pmtu: int | None = None
    reorder: int | None = None


class IntervalSum(BaseModel):
    start: float
    end: float
    seconds: float
    bytes: int
    bits_per_second: float
    omitted: bool
    sender: bool
    retransmits: int | None = None

    @property
    def throughput_gbps(self) -> float:
        """Throughput in Gbps."""
        return self.bits_per_second / 1e9

    @property
    def data_mb(self) -> float:
        """Data transferred in MB."""
        return self.bytes / 1e6

    @property
    def time(self) -> float:
        """Alias for start time."""
        return self.start


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

    @property
    def throughput_gbps(self) -> float:
        """Throughput in Gbps."""
        return self.bits_per_second / 1e9

    @property
    def data_mb(self) -> float:
        """Data transferred in MB."""
        return self.bytes / 1e6


class CpuUtilization(BaseModel):
    host_total: float
    host_user: float
    host_system: float
    remote_total: float
    remote_user: float
    remote_system: float


class StreamSender(StreamBase):
    pass


class StreamReceiver(StreamBase):
    pass


class EndStream(BaseModel):
    sender: StreamSender
    receiver: StreamReceiver


class EndSection(BaseModel):
    streams: list[EndStream]
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


class Connected(BaseModel):
    socket: int
    local_host: str
    local_port: int
    remote_host: str
    remote_port: int


class TimeStamp(BaseModel):
    time: str
    timesecs: int
    timemillisecs: int | None = None


class ConnectingTo(BaseModel):
    host: str
    port: int


class StartSection(BaseModel):
    connected: list[Connected]
    version: str
    system_info: str
    timestamp: TimeStamp
    connecting_to: ConnectingTo
    cookie: str
    tcp_mss: int | None = None
    tcp_mss_default: int | None = None
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


class IperfResult(BaseModel):
    run_id: str
    start_time: datetime
    end_time: datetime | None = None
    config: IperfClientConfig | IperfServerConfig
    general_config: GeneralConfig
    node_info: NodeInfo = Field(default_factory=NodeInfo)
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    json_results: Iperf3JsonResult | None = None
    output_directory: Path
    stdout_file: Path | None = None
    stderr_file: Path | None = None
    json_file: Path | None = None

    def coerce_full_paths(self):
        path_attrs = ["output_directory", "stdout_file", "stderr_file", "json_file"]
        for attr_name in path_attrs:
            path = getattr(self, attr_name)
            if path and isinstance(path, Path):
                setattr(self, attr_name, path.resolve(strict=False))
        return self

    def save(self) -> None:
        directory = self.output_directory
        directory.mkdir(parents=True, exist_ok=True)
        self.stderr_file = directory / STDERR_FILENAME
        self._write_if_content(self.stderr_file, self.stderr)
        if self.json_results:
            self.json_file = directory / JSON_RESULTS_FILENAME
            self._write_json(self.json_file, self.json_results.model_dump())
        self.stdout_file = directory / STDOUT_FILENAME
        self._write_if_content(self.stdout_file, self.stdout)

        self.coerce_full_paths()
        self._write_json(
            directory / METADATA_FILENAME,
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
