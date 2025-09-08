import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .models import (
    GeneralConfig,
    Iperf3JsonResult,
    IperfClientConfig,
    IperfResult,
    IperfServerConfig,
)


class Iperf3Runner:
    """Main class for running iperf3 commands and processing results."""

    def __init__(self, iperf3_path: Path):
        self.iperf3_path = str(iperf3_path)
        self.console = Console()
        try:
            self._validate_iperf3_availability()
        except RuntimeError:
            try:
                self.iperf3_path = str(iperf3_path.resolve())
                self._validate_iperf3_availability()
            except RuntimeError as e:
                self.console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1) from e

    def build_command(
        self,
        config: IperfClientConfig | IperfServerConfig,
        general_config: GeneralConfig,
    ) -> list[str]:
        cmd: list[str] = []
        if general_config.numa_node:
            cmd.extend(["numactl", f"--cpunodebind={general_config.numa_node}"])
        cmd.extend([self.iperf3_path])
        if isinstance(config, IperfClientConfig):
            if not config.server_host:
                raise ValueError("Client mode requires server_host to be set")
            cmd.extend(["-c", config.server_host])
        elif isinstance(config, IperfServerConfig):
            cmd.extend(["-s"])
        cmd.extend(config.build_cli_args())
        return cmd

    def run(
        self,
        config: IperfClientConfig | IperfServerConfig,
        general_config: GeneralConfig,
        timeout: int | None = None,
    ) -> IperfResult | None:
        # Determine the description based on config type
        if isinstance(config, IperfClientConfig):
            description = "Running iperf3 client test..."
            command_file = "client_command.txt"
        else:
            description = "Running iperf3 server..."
            command_file = "server_command.txt"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(description, total=None)
            try:
                run_id = general_config.run_id or str(uuid.uuid4())
                output_dir = self._create_output_directory(general_config, run_id)
                cmd = self.build_command(config, general_config)
                # Save command to file
                (output_dir / command_file).write_text(" ".join(cmd))
                start_time = datetime.now(timezone.utc)

                process_result = self._run_subprocess(
                    cmd, cwd=output_dir, timeout=timeout
                )
                json_results = None
                if isinstance(config, IperfServerConfig):
                    return
                try:
                    json_results = Iperf3JsonResult.model_validate_json(
                        process_result.stdout
                    )
                except ValidationError as e:
                    self.console.print(f"[red]Error parsing JSON output: {e}.[/red]")
                result = IperfResult(
                    run_id=run_id,
                    start_time=start_time,
                    config=config,
                    output_directory=output_dir,
                    stdout=process_result.stdout,
                    stderr=process_result.stderr,
                    return_code=process_result.returncode,
                    end_time=datetime.now(timezone.utc),
                    general_config=general_config,
                    json_results=json_results,
                )

                result.save()
                progress.update(task, description="✅ Completed")
                return result
            except Exception as e:
                progress.update(task, description=f"❌ Failed: {e}")
                self.console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1) from e

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
    def _create_output_directory(general_config: GeneralConfig, run_id: str) -> Path:
        base_dir = general_config.output_directory
        if general_config.results_dir_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            dir_name = f"{timestamp}_{run_id}"
        else:
            dir_name = run_id
        output_dir = base_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
