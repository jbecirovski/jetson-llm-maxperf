#!/usr/bin/env python3
"""Aggregate energy.json files from the energy campaign into the final grid.

Groups runs by (power mode, variant) from the folder name
(<date>-energy-<mode>-<variant>-<nn>), reports per-group medians and the
min-max spread, plus the input coefficient for long-context runs.

Usage: python3 aggregate-energy.py [benchmarks_dir]   (default: benchmarks)
Skips folders with SMOKE in the name (validation runs, out of protocol).
"""
import glob
import json
import os
import re
import sys


def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "benchmarks"
    groups = {}
    for path in sorted(glob.glob(os.path.join(root, "*energy*", "energy.json"))):
        folder = os.path.basename(os.path.dirname(path))
        if "SMOKE" in folder:
            continue
        m = re.match(r".*-energy-([a-z0-9]+)-([a-z0-9]+)-\d+$", folder)
        if not m:
            continue
        groups.setdefault((m.group(1), m.group(2)), []).append(
            json.load(open(path)))

    print(f"{'mode':6s} {'variant':8s} {'n':>2s} {'tok/s':>6s} {'W dec':>6s} "
          f"{'J/tok':>6s} {'Wh/Mtok':>8s} {'spread':>7s} {'idle W':>6s} "
          f"{'Wh/Mtok IN':>10s}")
    for (mode, variant), runs in sorted(groups.items()):
        wh = [r["wh_per_mtok"] for r in runs]
        row = {
            "mode": mode, "variant": variant, "n_runs": len(runs),
            "tok_per_s_median": round(median([r["tok_per_s"] for r in runs]), 1),
            "decode_w_median": round(median([r["decode_mean_w"] for r in runs]), 2),
            "j_per_token_median": round(median([r["j_per_token"] for r in runs]), 3),
            "wh_per_mtok_median": round(median(wh), 1),
            "wh_per_mtok_min": min(wh), "wh_per_mtok_max": max(wh),
            "idle_w_median": round(median([r["idle_pre_w"] for r in runs]), 2),
        }
        win = [r["wh_per_mtok_input"] for r in runs if r.get("wh_per_mtok_input")]
        if win:
            row["wh_per_mtok_input_median"] = round(median(win), 1)
            row["prefill_w_median"] = round(
                median([r["prefill_mean_w"] for r in runs if r.get("prefill_mean_w")]), 2)
            row["prefill_tok_per_s_median"] = round(median(
                [r["prefill_tokens"] / r["prefill_s"] for r in runs
                 if r.get("prefill_s")]), 0)
        print(f"{mode:6s} {variant:8s} {len(runs):2d} "
              f"{row['tok_per_s_median']:6.1f} {row['decode_w_median']:6.2f} "
              f"{row['j_per_token_median']:6.3f} {row['wh_per_mtok_median']:8.1f} "
              f"{max(wh) - min(wh):7.1f} {row['idle_w_median']:6.2f} "
              f"{row.get('wh_per_mtok_input_median', float('nan')):10.1f}")
        groups[(mode, variant)] = row

    out = {"%s-%s" % k: v for k, v in groups.items()}
    with open(os.path.join(root, "energy-summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten: {os.path.join(root, 'energy-summary.json')}")


if __name__ == "__main__":
    main()
