# bench/ - the common harness

Two scripts shared by every campaign. They are the reason a number in this
repo can be traced back to the exact machine state that produced it.

| Script | Role |
| --- | --- |
| `capture-jetson-env.sh` | Snapshots the full benchmark-relevant board configuration as JSON, and optionally wraps a command so its environment, command line, output and exit code are saved together. |
| `maximize-perf.sh` | Puts the board in MAXN and locks the clocks (`jetson_clocks`), the reproducible high-performance state used by the MAXN blocks. |

## Usage

```bash
./capture-jetson-env.sh                      # print the snapshot to stdout
./capture-jetson-env.sh -o env.json          # write it to a file
./capture-jetson-env.sh --wrap -- <command>  # run a benchmark and save the proof
```

The `--wrap` form creates a `bench-run-<timestamp>/` directory containing
`env.json`, `command.txt`, `stdout.log`, `stderr.log` and `exit-code.txt`.
Campaign scripts call it for every run, then rename the directory to a
meaningful name.

Proof directories are meant to be committed, so `command.txt` normalizes the
home directory to `/home/user`. A run has to be identifiable; the person who
launched it does not.

```bash
sudo ./maximize-perf.sh    # MAXN + locked clocks
```

**After locking the clocks, a reboot is required before changing power mode.**
`jetson_clocks` barely changes throughput (+0.7% measured on 2026-08-05) but
divides the standard deviation by 16 - it is used for reproducibility, not for
speed.

## Reading the power mode

```bash
nvpmodel -q                    # current mode
echo YES | sudo nvpmodel -m 0  # MAXN
echo YES | sudo nvpmodel -m 2  # 30 W
```
