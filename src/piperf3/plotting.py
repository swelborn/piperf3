"""Plotting utilities for iperf3 results visualization."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .models import Iperf3JsonResult, IperfResult


class IperfPlotter:
    """Class for creating plots from iperf3 results."""

    def __init__(
        self,
        style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "white",
    ):
        sns.set_style(style)
        plt.style.use("seaborn-v0_8-muted")
        self.colors = sns.color_palette("muted", 16)

    # -------------------------
    # Public Plot Methods
    # -------------------------

    def plot_multi_stream_comparison(
        self, result: IperfResult, output_file: Path | None = None
    ):
        """Plot stacked cumulative throughput of multiple parallel streams."""

        if not result.json_results:
            raise ValueError("JSON results required for multi-stream plotting")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Collect stream data for plotting
        stream_data = {}
        all_throughputs = []
        stream_labels = []

        for interval_idx, interval in enumerate(result.json_results.intervals):
            # Check if this is the first interval to validate interval duration
            if interval_idx == 0:
                interval_duration = (
                    interval.sum.seconds if hasattr(interval, "sum") else 1.0
                )
                if abs(interval_duration - 1.0) > 0.1:  # More than 100ms difference
                    print(
                        f"Warning: Intervals are {interval_duration:.3f}s, not 1s. Time axis may be misleading."
                    )

            for stream in interval.streams:
                sid = stream.socket
                if sid not in stream_data:
                    stream_data[sid] = {"times": [], "throughputs": []}

                # Use interval index + 1 as the time to show data at interval endpoints
                # This gives us time points: 1, 2, 3, 4, 5, etc.
                aligned_time = interval_idx + 1
                stream_data[sid]["times"].append(aligned_time)
                stream_data[sid]["throughputs"].append(stream.throughput_gbps)

                # Collect for distribution plot
                all_throughputs.append(stream.throughput_gbps)
                stream_labels.append(f"Stream {sid}")

        # Prepare data for stacked area plot
        if stream_data:
            # Get common time points (assuming all streams have same time intervals)
            times = next(iter(stream_data.values()))["times"]

            # Create matrix of throughput values for each stream at each time
            throughput_matrix = []
            stream_ids = list(stream_data.keys())

            for sid in stream_ids:
                throughput_matrix.append(stream_data[sid]["throughputs"])

            # Create stacked area plot
            throughput_array = np.array(throughput_matrix)

            ax1.stackplot(
                times,
                *throughput_array,
                labels=[f"Stream {sid}" for sid in stream_ids],
                colors=self.colors[: len(stream_ids)],
                alpha=0.8,
            )

            # Add total throughput line
            total_throughput = np.sum(throughput_array, axis=0)
            ax1.plot(times, total_throughput, "k-", linewidth=2, label="Total")

            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Cumulative Throughput (Gbps)")
            ax1.set_title("Stacked Stream Throughput Over Time")

        # Distribution of throughput per stream
        df = pd.DataFrame(
            {"Throughput (Gbps)": all_throughputs, "Stream": stream_labels}
        )

        # Create box plot with matching colors
        if stream_data:
            # Get unique stream IDs to match colors with stacked plot
            unique_streams = list(stream_data.keys())
            stream_colors = {
                f"Stream {sid}": self.colors[i] for i, sid in enumerate(unique_streams)
            }

            # Create color palette for box plot matching the stacked plot
            box_colors = [
                stream_colors.get(stream, self.colors[0])
                for stream in df["Stream"].unique()
            ]

            sns.boxplot(
                data=df,
                x="Stream",
                y="Throughput (Gbps)",
                hue="Stream",
                ax=ax2,
                palette=box_colors,
                legend=False,
            )
        else:
            sns.boxplot(data=df, x="Stream", y="Throughput (Gbps)", ax=ax2)

        ax2.set_title("Throughput Distribution by Stream")
        ax2.tick_params(axis="x", rotation=45)

        self._finalize_figure(fig, output_file)
        return fig

    def plot_comparison(
        self,
        results: list[IperfResult],
        output_file: Path | None = None,
        title: str = "Iperf3 Results Comparison",
    ):
        """Compare throughput and data transfer between multiple runs."""
        if not results:
            raise ValueError("At least one result is required")

        df = self._build_comparison_dataframe(results)
        if df.empty:
            raise ValueError("No valid JSON results found for comparison")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        x_pos = np.arange(len(df))
        width = 0.35

        self._plot_bar_comparison(
            ax1,
            x_pos,
            df["sent_gbps"],
            df["received_gbps"],
            "Throughput (Gbps)",
            "Throughput Comparison",
            df,
            width,
            self.colors[0],
            self.colors[1],
        )
        self._plot_bar_comparison(
            ax2,
            x_pos,
            df["sent_mb"],
            df["received_mb"],
            "Data Transferred (MB)",
            "Data Transfer Comparison",
            df,
            width,
            self.colors[2],
            self.colors[3],
        )
        ax3.bar(x_pos, df["duration"], color=self.colors[4])
        ax3.set_xlabel("Test Runs")
        ax3.set_ylabel("Duration (seconds)")
        ax3.set_title("Test Duration Comparison")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(
            [row["run_id"] for _, row in df.iterrows()],
            rotation=45,
            ha="right",
        )
        ax3.grid(True, alpha=0.3)

        ax4.axis("off")
        fig.suptitle(title, fontsize=16, fontweight="bold")
        self._finalize_figure(fig, output_file)
        return fig

    # -------------------------
    # Data Extraction Helpers
    # -------------------------
    @staticmethod
    def _require_json_results(result: IperfResult) -> None:
        if not result.json_results:
            raise ValueError("JSON results required for plotting")

    @staticmethod
    def _extract_time_series(result: IperfResult):
        if not result.json_results:
            raise ValueError("JSON results required for time series extraction")
        times, throughput_gbps, data_mb = [], [], []
        for interval in result.json_results.intervals:
            sum_data = interval.sum
            times.append(sum_data.time)
            throughput_gbps.append(sum_data.throughput_gbps)
            data_mb.append(sum_data.data_mb)
        return times, throughput_gbps, data_mb

    @staticmethod
    def _build_comparison_dataframe(results: list[IperfResult]) -> pd.DataFrame:
        comparison_data = []
        for res in results:
            if res.json_results:
                sent = res.json_results.end.sum_sent
                recv = res.json_results.end.sum_received
                comparison_data.append(
                    {
                        "run_id": res.run_id[:8],
                        "sent_gbps": sent.throughput_gbps,
                        "received_gbps": recv.throughput_gbps,
                        "sent_mb": sent.data_mb,
                        "received_mb": recv.data_mb,
                        "duration": sent.seconds,
                    }
                )
        return pd.DataFrame(comparison_data)

    # -------------------------
    # Plotting Helpers
    # -------------------------
    @staticmethod
    def _plot_line(ax, x, y, xlabel, ylabel, title, color, marker, label=None):
        ax.plot(x, y, marker=marker, color=color, linewidth=2, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _plot_bar_comparison(
        self,
        ax,
        x_pos,
        sent_data,
        recv_data,
        ylabel,
        title,
        df,
        width,
        sent_color,
        recv_color,
    ):
        ax.bar(x_pos - width / 2, sent_data, width, label="Sent", color=sent_color)
        ax.bar(x_pos + width / 2, recv_data, width, label="Received", color=recv_color)
        ax.set_xlabel("Test Runs")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [row["run_id"] for _, row in df.iterrows()],
            rotation=45,
            ha="right",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _finalize_figure(fig, output_file: Path | None):
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")

    # -------------------------
    # Utility Methods
    # -------------------------
    def _extract_summary_stats(self, json_results: Iperf3JsonResult) -> dict[str, str]:
        sent = json_results.end.sum_sent
        recv = json_results.end.sum_received
        return {
            "Start Time": json_results.start.timestamp.time,
            "Duration": f"{sent.seconds:.2f} sec",
            "Protocol": json_results.start.test_start.protocol,
            "Sent Throughput": f"{sent.throughput_gbps:.2f} Gbps",
            "Recv Throughput": f"{recv.throughput_gbps:.2f} Gbps",
            "Sent Data": f"{sent.data_mb:.2f} MB",
            "Recv Data": f"{recv.data_mb:.2f} MB",
        }

    @staticmethod
    def save_results_csv(results: list[IperfResult], output_file: Path) -> None:
        csv_data = []
        for result in results:
            row = {
                "run_id": result.run_id,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "return_code": result.return_code,
                "output_directory": str(result.output_directory),
            }
            if result.json_results:
                sent = result.json_results.end.sum_sent
                recv = result.json_results.end.sum_received
                row.update(
                    {
                        "sent_bits_per_second": sent.bits_per_second,
                        "sent_bytes": sent.bytes,
                        "sent_seconds": sent.seconds,
                        "received_bits_per_second": recv.bits_per_second,
                        "received_bytes": recv.bytes,
                        "received_seconds": recv.seconds,
                    }
                )
            csv_data.append(row)
        pd.DataFrame(csv_data).to_csv(output_file, index=False)
