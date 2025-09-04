# piperf3

A mostly vibe-coded wrapper around `iperf3` for running, collecting, and plotting network performance tests. Very unstable, probably only works with `iperf v3.19`.

## Requirements

`iperf3` binary in path. You should be able to run `iperf3` from command line.

## Installation

```bash
pip install piperf3
```

## CLI usage

- `piperf3 client SERVER [OPTIONS]` — run a client test with `SERVER` (hostname/IP)
- `piperf3 server [OPTIONS]` — run an iperf3 server

## License

MIT — see the `LICENSE` file.
