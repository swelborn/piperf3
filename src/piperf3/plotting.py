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
        style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "darkgrid",
    ):
        sns.set_style(style)
        plt.style.use("seaborn-v0_8-darkgrid")
        self.colors = sns.color_palette("husl", 10)

    # -------------------------
    # Public Plot Methods
    # -------------------------
    def plot_throughput_time_series(
        self,
        result: IperfResult,
        output_file: Path | None = None,
        title: str | None = None,
    ):
        """Plot throughput over time and cumulative data transfer."""
        self._require_json_results(result)

        times, throughput_bits, throughput_bytes = self._extract_time_series(result)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self._plot_line(
            ax1,
            times,
            throughput_bits,
            "Time (seconds)",
            "Throughput (Gbps)",
            "Throughput vs Time",
            self.colors[0],
            "o",
        )
        cumulative_bytes = np.cumsum(throughput_bytes)
        self._plot_line(
            ax2,
            times,
            cumulative_bytes,
            "Time (seconds)",
            "Cumulative Data (MB)",
            "Cumulative Data Transfer",
            self.colors[1],
            "s",
        )

        fig.suptitle(
            title or f"Iperf3 Results - {result.environment_name}",
            fontsize=16,
            fontweight="bold",
        )
        self._finalize_figure(fig, output_file)
        return fig

    def plot_multi_stream_comparison(
        self, result: IperfResult, output_file: Path | None = None
    ):
        """Plot comparison of multiple parallel streams."""
        self._require_json_results(result)

        stream_data = self._extract_stream_data(result)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Individual stream throughput
        for i, (sid, data) in enumerate(stream_data.items()):
            self._plot_line(
                ax1,
                data["times"],
                data["throughput"],
                "Time (seconds)",
                "Throughput (Gbps)",
                "Individual Stream Throughput",
                self.colors[i % len(self.colors)],
                "o",
                label=f"Stream {sid}",
            )
        ax1.legend()

        # Distribution of throughput per stream
        all_throughputs = [
            val for data in stream_data.values() for val in data["throughput"]
        ]
        stream_labels = [
            f"Stream {sid}"
            for sid, data in stream_data.items()
            for _ in data["throughput"]
        ]
        df = pd.DataFrame(
            {"Throughput (Gbps)": all_throughputs, "Stream": stream_labels}
        )
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
            [f"{row['name']}\n{row['run_id']}" for _, row in df.iterrows()],
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
        times, throughput_bits, throughput_bytes = [], [], []
        for interval in result.json_results.intervals:
            sum_data = interval.sum
            times.append(sum_data.start)
            throughput_bits.append(sum_data.bits_per_second / 1e9)  # Gbps
            throughput_bytes.append(sum_data.bytes / 1e6)  # MB
        return times, throughput_bits, throughput_bytes

    @staticmethod
    def _extract_stream_data(result: IperfResult):
        stream_data = {}
        for interval in result.json_results.intervals:
            time = interval.sum.start
            for stream in interval.streams:
                sid = stream.socket
                stream_data.setdefault(sid, {"times": [], "throughput": []})
                stream_data[sid]["times"].append(time)
                stream_data[sid]["throughput"].append(stream.bits_per_second / 1e9)
        return stream_data

    @staticmethod
    def _build_comparison_dataframe(results: list[IperfResult]) -> pd.DataFrame:
        comparison_data = []
        for res in results:
            if res.json_results:
                sent = res.json_results.end.sum_sent
                recv = res.json_results.end.sum_received
                comparison_data.append(
                    {
                        "name": res.environment_name,
                        "run_id": res.run_id[:8],
                        "sent_gbps": sent.bits_per_second / 1e9,
                        "received_gbps": recv.bits_per_second / 1e9,
                        "sent_mb": sent.bytes / 1e6,
                        "received_mb": recv.bytes / 1e6,
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
            [f"{row['name']}\n{row['run_id']}" for _, row in df.iterrows()],
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
            "Start Time": json_results.start.timestamp["time"],
            "Duration": f"{sent.seconds:.2f} sec",
            "Protocol": json_results.start.test_start.protocol,
            "Sent Throughput": f"{sent.bits_per_second / 1e9:.2f} Gbps",
            "Recv Throughput": f"{recv.bits_per_second / 1e9:.2f} Gbps",
            "Sent Data": f"{sent.bytes / 1e6:.2f} MB",
            "Recv Data": f"{recv.bytes / 1e6:.2f} MB",
        }

    @staticmethod
    def save_results_csv(results: list[IperfResult], output_file: Path) -> None:
        csv_data = []
        for result in results:
            row = {
                "environment_name": result.environment_name,
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
