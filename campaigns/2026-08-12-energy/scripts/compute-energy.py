#!/usr/bin/env python3
"""Compute energy-per-token from a proof folder produced by benchmark-energy.sh.

Inputs (inside the run directory):
    power.csv    - INA3221 samples (power-sampler.py)
    markers.txt  - wall-clock phase markers (idle_pre_start, cmd_start,
                   cmd_end, idle_post_end)
    stderr.log   - llama-speculative output (token counts and phase durations)

Windows:
    idle_pre  = [idle_pre_start, cmd_start]   -> median idle power
    idle_post = [cmd_end, idle_post_end]      -> median idle power (drift check)
    command   = [cmd_start, cmd_end]          -> includes model load, NOT used
                                                 for J/token
    decode    = the last `decoded ... in B s` seconds of the command window,
                anchored at cmd_end minus TAIL_S (process teardown after the
                perf printout). J/token integrates THIS window only: model
                load and prefill are excluded (prefill is reported apart).

Energy = trapezoidal integral of total_mw over the window.
Outputs energy.json in the run directory and a one-line summary on stdout.

Env: TAIL_S (default 1.0) - seconds between end of decoding and process exit.
Sanity signals in the JSON: sample_count, max_gap_s, and the mean decode power
vs command power (the decode plateau should be the highest sustained phase).
"""
import json
import os
import re
import sys


def load_markers(path):
    m = {}
    for line in open(path):
        k, v = line.split()
        m[k] = float(v)
    for k in ("idle_pre_start", "cmd_start", "cmd_end", "idle_post_end"):
        if k not in m:
            raise SystemExit(f"marker missing: {k}")
    return m


def load_samples(path):
    ts, mw = [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5:
                ts.append(float(parts[0]))
                mw.append(float(parts[4]))
    if len(ts) < 10:
        raise SystemExit(f"too few power samples: {len(ts)}")
    return ts, mw


def window(ts, mw, t0, t1):
    pairs = [(t, p) for t, p in zip(ts, mw) if t0 <= t <= t1]
    if len(pairs) < 3:
        raise SystemExit(f"too few samples in window [{t0:.2f},{t1:.2f}]: {len(pairs)}")
    return pairs


def trapz_joules(pairs):
    e = 0.0
    for (t0, p0), (t1, p1) in zip(pairs, pairs[1:]):
        e += (p0 + p1) / 2.0 * (t1 - t0)
    return e / 1000.0  # mW*s -> J


def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def parse_stderr(path):
    """Token counts and phase durations from llama-speculative output."""
    txt = open(path, errors="replace").read()
    out = {}
    m = re.search(r"encoded\s+(\d+)\s+tokens?\s+in\s+([\d.]+)\s+seconds", txt)
    if m:
        out["prefill_tokens"], out["prefill_s"] = int(m.group(1)), float(m.group(2))
    m = re.search(r"decoded\s+(\d+)\s+tokens?\s+in\s+([\d.]+)\s+seconds", txt)
    if m:
        out["decoded_tokens"], out["decode_s"] = int(m.group(1)), float(m.group(2))
    for key in ("n_predict", "n_drafted", "n_accept"):
        m = re.search(rf"{key}\s*=\s*(\d+)", txt)
        if m:
            out[key] = int(m.group(1))
    if "decoded_tokens" not in out:
        raise SystemExit("could not parse 'decoded N tokens in S seconds' from stderr.log")
    return out


def main():
    run_dir = sys.argv[1]
    tail_s = float(os.environ.get("TAIL_S", "1.0"))
    mk = load_markers(os.path.join(run_dir, "markers.txt"))
    ts, mw = load_samples(os.path.join(run_dir, "power.csv"))
    st = parse_stderr(os.path.join(run_dir, "stderr.log"))

    idle_pre_mw = median([p for _, p in window(ts, mw, mk["idle_pre_start"], mk["cmd_start"])])
    idle_post_mw = median([p for _, p in window(ts, mw, mk["cmd_end"], mk["idle_post_end"])])

    cmd_pairs = window(ts, mw, mk["cmd_start"], mk["cmd_end"])
    e_cmd_j = trapz_joules(cmd_pairs)

    t_dec_end = mk["cmd_end"] - tail_s
    t_dec_start = t_dec_end - st["decode_s"]
    dec_pairs = window(ts, mw, t_dec_start, t_dec_end)
    e_dec_j = trapz_joules(dec_pairs)
    dec_mean_mw = e_dec_j * 1000.0 / st["decode_s"]

    # Prefill (input) coefficient: the window right before decoding. Only
    # meaningful on long-context runs; with a tiny prompt (<1 s) there are too
    # few samples and the numbers stay None. This is the second coefficient of
    # the robot budget: E = wh_per_mtok_input x input + wh_per_mtok x output.
    e_pre_j = wh_in = pre_mean_w = None
    if st.get("prefill_s") and st.get("prefill_tokens"):
        try:
            pre_pairs = window(ts, mw, t_dec_start - st["prefill_s"], t_dec_start)
            e_pre_j = trapz_joules(pre_pairs)
            pre_mean_w = round(e_pre_j / st["prefill_s"], 2)
            wh_in = round(e_pre_j / st["prefill_tokens"] * 1e6 / 3600.0, 1)
            e_pre_j = round(e_pre_j, 2)
        except SystemExit:
            pass

    n = st["decoded_tokens"]
    j_per_tok = e_dec_j / n
    j_per_tok_marginal = (e_dec_j - idle_pre_mw / 1000.0 * st["decode_s"]) / n

    gaps = [t1 - t0 for (t0, _), (t1, _) in zip(cmd_pairs, cmd_pairs[1:])]
    result = {
        "decoded_tokens": n,
        "decode_s": st["decode_s"],
        "tok_per_s": n / st["decode_s"],
        "energy_decode_j": round(e_dec_j, 2),
        "j_per_token": round(j_per_tok, 4),
        "j_per_token_marginal": round(j_per_tok_marginal, 4),
        "wh_per_mtok": round(j_per_tok * 1e6 / 3600.0, 1),
        "wh_per_mtok_marginal": round(j_per_tok_marginal * 1e6 / 3600.0, 1),
        "decode_mean_w": round(dec_mean_mw / 1000.0, 2),
        "idle_pre_w": round(idle_pre_mw / 1000.0, 2),
        "idle_post_w": round(idle_post_mw / 1000.0, 2),
        "energy_cmd_total_j": round(e_cmd_j, 2),
        "cmd_s": round(mk["cmd_end"] - mk["cmd_start"], 2),
        "prefill_tokens": st.get("prefill_tokens"),
        "prefill_s": st.get("prefill_s"),
        "energy_prefill_j": e_pre_j,
        "prefill_mean_w": pre_mean_w,
        "wh_per_mtok_input": wh_in,
        "n_drafted": st.get("n_drafted"),
        "n_accept": st.get("n_accept"),
        "tail_s": tail_s,
        "sample_count": len(ts),
        "max_gap_s_in_cmd": round(max(gaps), 3) if gaps else None,
        "method": "trapezoidal integral of VDD_GPU_SOC+VDD_CPU_CV+VIN_SYS_5V0 "
                  "(0x40 rails, VDDQ excluded: subset of VIN_SYS_5V0); decode "
                  "window anchored at cmd_end - tail_s; internal sensors = "
                  "lower bound vs wall (Padova: ~1.02x + 3.1 W)",
    }
    with open(os.path.join(run_dir, "energy.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("%s: %.1f tok/s, %.2f W decode, %.3f J/tok (%.0f Wh/Mtok), idle %.2f/%.2f W"
          % (os.path.basename(run_dir.rstrip("/")), result["tok_per_s"],
             result["decode_mean_w"], j_per_tok, result["wh_per_mtok"],
             result["idle_pre_w"], result["idle_post_w"]))


if __name__ == "__main__":
    main()
