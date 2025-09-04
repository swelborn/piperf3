from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import IperfResult

console = Console()


def display_results(result: IperfResult, verbose: bool = False):
    if result.return_code != 0:
        console.print(f"[red]Test failed with return code {result.return_code}[/red]")
        if result.stderr:
            console.print(f"[red]Error: {result.stderr}[/red]")
        return

    table = Table(title="Test Results Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

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


def display_config_panel(title: str, info: str, style: str = "bold blue"):
    console.print(Panel(info, title=f"[{style}]{title}[/{style}]", expand=False))
