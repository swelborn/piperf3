import inspect
from functools import wraps

import typer
from pydantic_settings import BaseSettings
from rich.console import Console

from .display import display_config_panel, display_results
from .models import (
    GeneralConfig,
    IperfClientConfig,
    IperfResult,
    IperfServerConfig,
    TyperOptionSchema,
)
from .plotting import IperfPlotter
from .runner import Iperf3Runner

app = typer.Typer(
    name="piperf3",
    help="Python wrapper for iperf3 with configuration management and plotting.",
    add_completion=False,
)

console = Console()


def cli_from_models(*model_classes: type[BaseSettings]):
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
                cli_meta = (
                    field_info.json_schema_extra.get("cli")
                    if field_info.json_schema_extra
                    else None
                )
                if not cli_meta:
                    continue
                option_model = TyperOptionSchema(**cli_meta)  # type: ignore
                option = typer.Option(
                    option_model.default,
                    *option_model.param_decls,
                    help=option_model.help,
                    min=option_model.min,
                    max=option_model.max,
                    parser=option_model.parser,
                )
                new_params.append(
                    inspect.Parameter(
                        field_name,
                        kind=inspect.Parameter.KEYWORD_ONLY,
                        default=option,
                        annotation=field_info.annotation,
                    )
                )

        @wraps(func)
        def wrapper(*args, **kwargs):
            configs = {}
            # Collect all field names from all model classes
            all_model_fields = set()
            for model_cls in model_classes:
                all_model_fields.update(model_cls.model_fields.keys())

            for model_cls in model_classes:
                # First create config from environment/defaults
                base_config = model_cls()

                # Then override with CLI arguments
                model_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k in model_cls.model_fields
                    and v is not None
                    and v != model_cls.model_fields[k].default
                }

                # Apply CLI overrides to the base config
                for key, value in model_kwargs.items():
                    setattr(base_config, key, value)

                configs[model_cls] = base_config

            # Remove all model field arguments from kwargs before passing to function
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k not in all_model_fields
            }

            return func(
                *args,
                config=configs.get(IperfClientConfig) or configs.get(IperfServerConfig),
                general_config=configs.get(GeneralConfig),
                **filtered_kwargs,
            )

        wrapper.__signature__ = sig.replace(parameters=new_params)  # pyright: ignore[reportAttributeAccessIssue]
        return wrapper

    return decorator


def _generate_plots(result: IperfResult):
    plotter = IperfPlotter()
    try:
        plot_dir = result.output_directory / "plots"
        plot_dir.mkdir(exist_ok=True)
        plotter.plot_multi_stream_comparison(result, plot_dir / "stream_comparison.png")
        console.print(f"[green]Plots saved to: {plot_dir}[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not generate plots: {e}[/yellow]")


@cli_from_models(IperfClientConfig, GeneralConfig)
def client(
    server=typer.Argument(
        ..., envvar="IPERF_CLIENT_SERVER_HOST", help="Server hostname or IP"
    ),
    config: IperfClientConfig = None,  # pyright: ignore[reportArgumentType]
    general_config: GeneralConfig = None,  # pyright: ignore[reportArgumentType]
):
    """Run iperf3 client with the specified configuration."""
    config.server_host = server

    # Display full configuration
    config_info = config.pretty_print()
    display_config_panel("Full Configuration", config_info)

    plot = True
    if plot and not config.json_output:
        console.print("[yellow]Enabling JSON output for plotting[/yellow]")
        config.json_output = True

    runner = Iperf3Runner(iperf3_path=general_config.iperf3_path)
    result = runner.run(config, general_config)
    if result is None:
        return
    display_results(result, verbose=config.verbose)

    if plot and result.json_results:
        _generate_plots(result)
    console.print(f"[green]Results saved to: {result.output_directory}[/green]")


@cli_from_models(IperfServerConfig, GeneralConfig)
def server(
    config: IperfServerConfig = None,  # pyright: ignore[reportArgumentType]
    general_config: GeneralConfig = None,  # pyright: ignore[reportArgumentType]
):
    """Run iperf3 server with the specified configuration."""
    runner = Iperf3Runner(iperf3_path=general_config.iperf3_path)
    config_info = config.pretty_print()
    display_config_panel("Server Configuration", config_info, style="bold green")
    try:
        runner.run(config, general_config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error running server: {e}[/red]")
        raise typer.Exit(1) from e


app.command()(client)
app.command()(server)
