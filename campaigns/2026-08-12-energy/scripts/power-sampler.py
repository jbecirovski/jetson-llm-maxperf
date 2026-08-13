#!/usr/bin/env python3
"""Sample the Jetson AGX Orin onboard INA3221 power rails to a CSV file.

Reads the three channels of the 0x40 INA3221 (VDD_GPU_SOC, VDD_CPU_CV,
VIN_SYS_5V0) directly from sysfs, plus VDDQ_VDD2_1V8AO from the 0x41 chip
for information only. VDDQ is NEVER added to the total: it is a subset of
VIN_SYS_5V0 (NVIDIA forum thread 223111), summing it would double-count DDR.

The module total is the sum of the three 0x40 rails. Caveat carried by the
whole campaign: internal rails measure downstream of the regulators; the
published AGX Orin calibration (Shalavi et al., Univ. of Padova) puts the DC
wall power at ~1.02 x sensors + 3.1 W. These numbers are a lower bound,
comparable between runs on the same board.

Usage:
    python3 power-sampler.py --out power.csv [--interval 0.1]

Stops cleanly on SIGTERM/SIGINT. One CSV row per sample:
    epoch_s,gpu_soc_mw,cpu_cv_mw,sys_5v0_mw,total_mw,vddq_mw
Sampling is drift-corrected (scheduled on the monotonic clock); the INA3221
hardware conversion cycle is ~6.6 ms, so 100 ms sampling is comfortably
above it (NVIDIA forum thread 378378).
"""
import argparse
import glob
import signal
import sys
import time


def find_hwmon(chip):
    dirs = glob.glob(f"/sys/bus/i2c/drivers/ina3221/{chip}/hwmon/hwmon*")
    if not dirs:
        raise SystemExit(f"no hwmon dir for INA3221 {chip}")
    return dirs[0]


def rail_paths(hwmon, wanted):
    """Map rail label -> (voltage_path, current_path) for the wanted labels."""
    out = {}
    for lab in glob.glob(hwmon + "/in*_label"):
        name = open(lab).read().strip()
        if name in wanted:
            idx = lab.rsplit("/in", 1)[1].split("_")[0]
            out[name] = (f"{hwmon}/in{idx}_input", f"{hwmon}/curr{idx}_input")
    missing = set(wanted) - set(out)
    if missing:
        raise SystemExit(f"rails not found in {hwmon}: {missing}")
    return out


def read_mw(paths):
    mv = int(open(paths[0]).read())
    ma = int(open(paths[1]).read())
    return mv * ma / 1000.0  # mV * mA -> mW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.1)
    args = ap.parse_args()

    main_rails = rail_paths(find_hwmon("1-0040"),
                            ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"])
    vddq = rail_paths(find_hwmon("1-0041"), ["VDDQ_VDD2_1V8AO"])["VDDQ_VDD2_1V8AO"]
    order = ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"]

    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))

    with open(args.out, "w", buffering=1) as f:
        f.write("epoch_s,gpu_soc_mw,cpu_cv_mw,sys_5v0_mw,total_mw,vddq_mw\n")
        next_t = time.monotonic()
        while not stop:
            mws = [read_mw(main_rails[r]) for r in order]
            q = read_mw(vddq)
            f.write("%.6f,%.1f,%.1f,%.1f,%.1f,%.1f\n"
                    % (time.time(), mws[0], mws[1], mws[2], sum(mws), q))
            next_t += args.interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:  # fell behind (shouldn't happen at 10 Hz): resync, don't burst
                next_t = time.monotonic()
    sys.exit(0)


if __name__ == "__main__":
    main()
