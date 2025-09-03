import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .models import (
    GeneralConfig,
    Iperf3JsonResult,
    IperfClientConfig,
    IperfResult,
    IperfServerConfig,
)


class Iperf3Runner:
    """Main class for running iperf3 commands and processing results."""

    def __init__(self, iperf3_path: str = "iperf3"):
        self.iperf3_path = iperf3_path
        self._validate_iperf3_availability()

    # -------------------------
    # Public API
    # -------------------------
    def build_command(self, config: IperfClientConfig | IperfServerConfig) -> list[str]:
        cmd = [self.iperf3_path]
        if isinstance(config, IperfClientConfig):
            if not config.server_host:
                raise ValueError("Client mode requires server_host to be set")
            cmd.extend(["-c", config.server_host])
        elif isinstance(config, IperfServerConfig):
            cmd.append("-s")
        cmd.extend(config.build_cli_args())
        return cmd

    def run(
        self,
        config: IperfClientConfig | IperfServerConfig,
        general_config: GeneralConfig,
        timeout: int | None = None,
    ) -> IperfResult:
        run_id = general_config.run_id or str(uuid.uuid4())
        output_dir = self._create_output_directory(general_config, run_id)
        cmd = self.build_command(config)
        self._save_command(output_dir, cmd)

        result = IperfResult(
            environment_name=general_config.name,
            run_id=run_id,
            start_time=datetime.now(timezone.utc),
            config_used=config,
            output_directory=output_dir,
            provenance=general_config.get_provenance(),
        )

        try:
            process_result = self._run_subprocess(cmd, cwd=output_dir, timeout=timeout)
            result.stdout, result.stderr, result.return_code = (
                process_result.stdout,
                process_result.stderr,
                process_result.returncode,
            )
            result.end_time = datetime.now(timezone.utc)

            self._parse_and_store_json_results(result, output_dir)

        except subprocess.TimeoutExpired:
            self._handle_timeout(result, timeout)
        except Exception as e:
            self._handle_unexpected_error(result, e)

        result.save_to_directory(output_dir)
        return result

    def run_server_background(
        self, config: IperfServerConfig, general_config: GeneralConfig
    ) -> tuple[subprocess.Popen, Path]:
        run_id = general_config.run_id or str(uuid.uuid4())
        output_dir = self._create_output_directory(general_config, run_id)
        cmd = self.build_command(config)
        self._save_command(output_dir, cmd)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=output_dir,
            env=os.environ,
        )
        return process, output_dir

    def get_version(self) -> str:
        try:
            return self._run_subprocess(
                [self.iperf3_path, "--version"], timeout=10
            ).stdout.strip()
        except Exception as e:
            return f"Error getting version: {e!s}"

    # -------------------------
    # Internal Helpers
    # -------------------------
    def _validate_iperf3_availability(self) -> None:
        try:
            result = self._run_subprocess([self.iperf3_path, "--version"], timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"iperf3 is not working: {result.stderr}")
        except FileNotFoundError as err:
            raise RuntimeError(f"iperf3 not found at: {self.iperf3_path}") from err
        except subprocess.TimeoutExpired as err:
            raise RuntimeError("iperf3 version check timed out") from err

    @staticmethod
    def _run_subprocess(
        cmd: list[str], cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=os.environ,
        )

    @staticmethod
    def _save_command(output_dir: Path, cmd: list[str]) -> None:
        (output_dir / "command.txt").write_text(" ".join(cmd))

    @staticmethod
    def _create_output_directory(general_config: GeneralConfig, run_id: str) -> Path:
        base_dir = general_config.output_directory or Path("./results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{general_config.name}_{run_id[:8]}_{timestamp}"
        output_dir = base_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _parse_and_store_json_results(result: IperfResult, output_dir: Path) -> None:
        try:
            parsed = Iperf3JsonResult.model_validate_json(result.stdout)
            result.json_results = parsed
            (output_dir / "results.json").write_text(result.stdout)
            result.provenance["summary_metrics"] = {
                "sent_mbps": parsed.end.sum_sent.bits_per_second / 1e6,
                "received_mbps": parsed.end.sum_received.bits_per_second / 1e6,
                "sent_bytes": parsed.end.sum_sent.bytes,
                "received_bytes": parsed.end.sum_received.bytes,
                "duration_seconds": parsed.end.sum_sent.seconds,
                "cpu_utilization": parsed.end.cpu_utilization_percent.model_dump(),
            }
        except ValidationError as e:
            result.stderr += f"\nInvalid iperf3 JSON format: {e}"

    @staticmethod
    def _handle_timeout(result: IperfResult, timeout: int | None) -> None:
        result.stderr = f"Command timed out after {timeout} seconds"
        result.return_code = -1
        result.end_time = datetime.now(timezone.utc)

    @staticmethod
    def _handle_unexpected_error(result: IperfResult, e: Exception) -> None:
        result.stderr = f"Unexpected error: {e!s}"
        result.return_code = -1
        result.end_time = datetime.now(timezone.utc)
