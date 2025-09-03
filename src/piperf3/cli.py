import contextlib
import inspect
from functools import wraps

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .models import (
    GeneralConfig,
    IperfClientConfig,
    IperfResult,
    IperfServerConfig,
)
from .plotting import IperfPlotter
from .runner import Iperf3Runner

app = typer.Typer(
    name="piperf3",
    help="Python wrapper for iperf3 with configuration management and plotting.",
    add_completion=False,
)

console = Console()


def cli_from_models(*model_classes):
    """Decorator to inject Typer Option fields from Pydantic models into CLI commands."""

    def decorator(func):
        sig = inspect.signature(func)
        new_params = [
            p
            for p in sig.parameters.values()
            if p.name not in ("config", "general_config")
        ]

        for model_cls in model_classes:
            for field_name, field_info in model_cls.model_fields.items():
                default = field_info.default
                if not isinstance(default, typer.models.OptionInfo):
                    continue
                new_params.append(
                    inspect.Parameter(
                        field_name,
                        kind=inspect.Parameter.KEYWORD_ONLY,
                        default=default,
                        annotation=field_info.annotation,
                    )
                )

        @wraps(func)
        def wrapper(*args, **kwargs):
            configs = {}
            for model_cls in model_classes:
                model_kwargs = {
                    k: v for k, v in kwargs.items() if k in model_cls.model_fields
                }
                for k in model_kwargs:
                    kwargs.pop(k)
                configs[model_cls] = model_cls(**model_kwargs)
            return func(
                *args,
                config=configs.get(IperfClientConfig) or configs.get(IperfServerConfig),
                general_config=configs.get(GeneralConfig),
                **kwargs,
            )

        wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore
        return wrapper

    return decorator


def _display_config_panel(title: str, info: str, style: str = "bold blue"):
    console.print(Panel(info, title=f"[{style}]{title}[/{style}]", expand=False))


def _run_with_progress(description: str, func, *args, **kwargs):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)
        try:
            result = func(*args, **kwargs)
            progress.update(task, description="✅ Completed")
            return result
        except Exception as e:
            progress.update(task, description=f"❌ Failed: {e}")
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e


def _display_results(result: IperfResult, verbose: bool = False):
    if result.return_code != 0:
        console.print(f"[red]Test failed with return code {result.return_code}[/red]")
        if result.stderr:
            console.print(f"[red]Error: {result.stderr}[/red]")
        return

    table = Table(title="Test Results Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Environment", result.environment_name)
    table.add_row("Run ID", result.run_id[:8])
    table.add_row("Start Time", str(result.start_time))
    table.add_row(
        "Duration",
        str(result.end_time - result.start_time) if result.end_time else "N/A",
    )
    table.add_row("Return Code", str(result.return_code))

    if result.json_results:
        sent = result.json_results.end.sum_sent
        recv = result.json_results.end.sum_received
        table.add_row("Sent Throughput", f"{sent.bits_per_second / 1e9:.2f} Gbps")
        table.add_row("Sent Data", f"{sent.bytes / 1e6:.2f} MB")
        table.add_row("Received Throughput", f"{recv.bits_per_second / 1e9:.2f} Gbps")
        table.add_row("Received Data", f"{recv.bytes / 1e6:.2f} MB")

    console.print(table)

    if verbose and result.stdout:
        console.print("\n[bold]Raw Output:[/bold]")
        console.print(Panel(result.stdout, title="stdout", border_style="blue"))

    if result.stderr:
        console.print("\n[bold]Errors/Warnings:[/bold]")
        console.print(Panel(result.stderr, title="stderr", border_style="red"))


def _generate_plots(result: IperfResult):
    plotter = IperfPlotter()
    try:
        plot_dir = result.output_directory / "plots"
        plot_dir.mkdir(exist_ok=True)
        plotter.plot_throughput_time_series(result).savefig(
            plot_dir / "throughput_timeseries.png", dpi=300, bbox_inches="tight"
        )
        with contextlib.suppress(ValueError):
            plotter.plot_multi_stream_comparison(result).savefig(
                plot_dir / "stream_comparison.png", dpi=300, bbox_inches="tight"
            )
        console.print(f"[green]Plots saved to: {plot_dir}[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not generate plots: {e}[/yellow]")


@cli_from_models(IperfClientConfig, GeneralConfig)
def client(
    server: str = typer.Argument(..., help="Server hostname or IP address"),
    config: IperfClientConfig = None,  # pyright: ignore[reportArgumentType]
    general_config: GeneralConfig = None,  # pyright: ignore[reportArgumentType]
):
    """Run iperf3 client with the specified configuration."""
    config.server_host = server
    plot = True
    if plot and not config.json_output:
        console.print("[yellow]Enabling JSON output for plotting[/yellow]")
        config.json_output = True

    config_info = (
        f"Server: {config.server_host}:{config.port}\n"
        f"Duration: {config.time or 10}s\n"
        f"Protocol: {config.format or 'tcp'}\n"
        f"Parallel streams: {config.parallel_streams or 1}"
    )
    _display_config_panel("Client Configuration", config_info)

    runner = Iperf3Runner()
    result = _run_with_progress(
        "Running iperf3 client test...", runner.run, config, general_config
    )
    _display_results(result, verbose=config.verbose)

    if plot and result.json_results:
        _generate_plots(result)
    console.print(f"[green]Results saved to: {result.output_directory}[/green]")


@cli_from_models(IperfServerConfig, GeneralConfig)
def server(
    config: IperfServerConfig = None,  # pyright: ignore[reportArgumentType]
    general_config: GeneralConfig = None,  # pyright: ignore[reportArgumentType]
):
    """Run iperf3 server with the specified configuration."""
    runner = Iperf3Runner()

    if config.daemon:
        info = f"[bold green]Starting iperf3 server in daemon mode[/bold green]\nPort: {config.port or 5201}\nBind: {config.bind_address or 'all interfaces'}"
        _display_config_panel("Server Configuration", info, style="bold green")
        try:
            process, output_dir = runner.run_server_background(config, general_config)
            console.print(f"[green]Server started with PID {process.pid}[/green]")
            console.print(f"[green]Output directory: {output_dir}[/green]")
            if output_dir:
                (output_dir / "server.pid").write_text(str(process.pid))
        except Exception as e:
            console.print(f"[red]Error starting server: {e}[/red]")
            raise typer.Exit(1) from e
    else:
        info = f"[bold green]Starting iperf3 server[/bold green]\nPort: {config.port or 5201}\nBind: {config.bind_address or 'all interfaces'}\nOne-off: {config.one_off}"
        _display_config_panel("Server Configuration", info, style="bold green")
        try:
            result = _run_with_progress(
                "Running iperf3 server...", runner.run, config, general_config
            )
            _display_results(result, verbose=config.verbose)
            console.print(f"[green]Results saved to: {result.output_directory}[/green]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Server interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Error running server: {e}[/red]")
            raise typer.Exit(1) from e


app.command()(client)
app.command()(server)
