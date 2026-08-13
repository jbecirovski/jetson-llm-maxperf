# Campaign 2026-08-12 - Energy per token on AGX Orin

**Question**: how much energy does one million tokens cost on an embedded LLM,
and which configuration is cheapest on battery?

40 protocol runs plus a smoke test, in 8 blocks of 5. Every run ships its
full proof: environment snapshot, command, logs, raw power samples, computed
energy.

## Results (medians of 5 runs per block)

Short context (187-token prompt, 256 generated):

| Config | tok/s | W decode | J/token | Wh/Mtok output |
| --- | --- | --- | --- | --- |
| 30 W "efficiency" mode, 8B Q4_K_M | 6.6 | 18.0 | 2.71 | **754** |
| MAXN, 8B Q4_K_M (paired baseline) | 22.0 | 47.6 | 2.17 | **602** |
| MAXN, 8B IQ4_XS | 26.8 | 48.3 | 1.80 | **501** |
| MAXN, 8B + 1B draft (speculative) | 46.8 | 42.0 | 0.90 | **249** |

Long context (15,718 tokens read, 192 generated) - gives the input
coefficient, and shows what a full KV cache does to the output cost:

| Config | tok/s | W decode | Wh/Mtok output | Wh/Mtok **input** | output vs input |
| --- | --- | --- | --- | --- | --- |
| 30 W, 8B Q4_K_M | 4.7 | 17.3 | 1025 | **22.2** | 46x |
| MAXN, 8B Q4_K_M | 11.6 | 37.8 | 910 | **17.3** | 53x |
| MAXN, 8B IQ4_XS | 12.7 | 37.2 | 812 | **16.1** | 50x |
| MAXN, speculative | 30.9 | 41.5 | 372 | **17.5** | 21x |

Idle: 11.2 W (MAXN), 8.8 W (30 W mode). Aggregated numbers:
`results/energy-summary.json`.

What the numbers say:

1. **Race-to-idle**: the "efficiency" mode draws 2.6x fewer watts but is 3.3x
   slower, so it costs **25% more energy per token**. Slowing down to save
   power is a losing trade on this board.
2. **Speculative decoding wins twice**: 2.1x faster AND 5.6 W lower during
   decode, so 2.4x less energy per token. Hypothesis (labelled as such, not
   verified mechanically): batched validation reads the 8B weights once for
   several tokens, so less memory traffic per token, and memory traffic drives
   power here. Datacenter papers often report no energy gain from a plain
   draft model - this is an edge-specific result.
3. **Input is nearly free, and flat**: reading context costs 16-18 Wh/Mtok on
   every MAXN config, because prefill saturates compute rather than memory
   bandwidth - speeding up generation does not make reading cheaper. Only the
   30 W mode pays more (22.2). Generating costs 14x to 53x more than reading.
4. **A loaded context makes every output token dearer**: with 16K tokens in
   the KV cache, generation halves at nearly constant power, so the output
   coefficient rises ~50% (249 to 372 Wh/Mtok speculative, 602 to 910
   baseline, 501 to 812 on IQ4_XS). An agent carrying a long history pays
   twice: once to read it, then on every token it produces.
5. **The worst edge configuration exceeds the datacenter reference**: 30 W
   mode with a 16K context lands at 1025 Wh/Mtok, above the ~970 measured on
   an A100 (arXiv 2310.03003). Comparing two different instruments deserves
   caution, but the direction holds: edge is not frugal by nature, the
   configuration decides.

Caveat carried by every number: onboard INA3221 rails measure downstream of
the regulators, so these are a **lower bound**. Published AGX Orin
calibration (Shalavi et al., Univ. of Padova) puts DC wall power at
`1.02 x sensors + 3.1 W`. Numbers compare to each other on this board, not to
a wall meter.

## Scripts and how to use them

All scripts live in `scripts/` and run **on the board**, from a working
directory that also contains `bench/capture-jetson-env.sh` (the proof wrapper)
and the two prompt files.

| Script | Role |
| --- | --- |
| `power-sampler.py` | Samples the INA3221 rails to CSV. Standalone: `python3 power-sampler.py --out power.csv [--interval 0.1]`. Reads the three channels of the `1-0040` chip (VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0) from sysfs and sums them. **VDDQ_VDD2_1V8AO is logged but never summed**: it is a subset of VIN_SYS_5V0 (NVIDIA forum thread 223111), adding it double-counts DRAM. Stops cleanly on SIGTERM. |
| `benchmark-energy.sh` | Runs one block: for each run, 60 s idle, generation under the proof wrapper, 60 s idle, 30 s cooldown, then computes energy. Usage: `./benchmark-energy.sh <variant> [runs=5]`. |
| `compute-energy.py` | Post-processes one run directory into `energy.json`. Usage: `python3 compute-energy.py <run-dir>`. Called automatically by `benchmark-energy.sh`. |
| `aggregate-energy.py` | Medians per block into `energy-summary.json`. Usage: `python3 aggregate-energy.py [runs-dir]`. |
| `make-longctx-prompt.py` | Regenerates the long-context prompt deterministically (seed 42): `python3 make-longctx-prompt.py > longctx-prompt.txt`. The generated file is committed as `longctx-prompt.txt`; the script documents its provenance. |
| `energy-prompt.txt` / `longctx-prompt.txt` | The two prompts. Short (187 tokens) and long (15,718 tokens of synthetic warehouse-robot inspection logs, ending with a fixed-schema JSON instruction). |

### Variants of `benchmark-energy.sh`

| Variant | Model | Draft | Context | Measures |
| --- | --- | --- | --- | --- |
| `q4km` | 8B Q4_K_M | disabled (`--spec-draft-n-max 0`) | 4096 | output cost, paired baseline |
| `iq4xs` | 8B IQ4_XS | disabled | 4096 | output cost of a smaller quant |
| `spec` | 8B Q4_K_M | 1B, `n-max 8` | 4096 | output cost with speculative decoding |
| `longctx` | 8B Q4_K_M | 1B, `n-max 8` | 20480 | input coefficient + output cost under load |
| `lcq4km` | 8B Q4_K_M | disabled | 20480 | input coefficient, baseline |
| `lciq4xs` | 8B IQ4_XS | disabled | 20480 | input coefficient, IQ4_XS |

Every variant loads the **same binary** (`llama-speculative`) with the draft
model present; only `--spec-draft-n-max` changes (0 disables drafting,
`n_drafted=0` in the stats). One variable, constant referential. This is why
the baseline reads 22.0 tok/s here and not the 27.1 tok/s that `llama-bench`
reports for `tg128` - a different instrument, both are true.

Long-context variants need `-b` to cover the whole prompt: `llama-speculative`
submits it in a single `llama_decode` call, otherwise
`GGML_ASSERT(n_tokens_all <= cparams.n_batch)` aborts the run.

### Reproducing the campaign

```bash
# MAXN blocks (locks the clocks; a reboot is required to change power mode after)
sudo ./bench/maximize-perf.sh
./benchmark-energy.sh q4km 5
./benchmark-energy.sh iq4xs 5
./benchmark-energy.sh spec 5
./benchmark-energy.sh longctx 5
./benchmark-energy.sh lcq4km 5
./benchmark-energy.sh lciq4xs 5

# 30 W blocks: reboot, switch mode, run stock (no clock locking)
sudo reboot
echo YES | sudo nvpmodel -m 2      # 0 = MAXN, 2 = 30 W
./benchmark-energy.sh q4km 5
./benchmark-energy.sh lcq4km 5

python3 aggregate-energy.py        # -> energy-summary.json
```

## Method

1. **Sampling**: three `1-0040` rails read from sysfs at 10 Hz (hardware
   conversion cycle is ~6.6 ms, so 100 ms is comfortably above it), scheduled
   on the monotonic clock so sampling does not drift.
2. **Windows**: 60 s idle before and after each generation. Energy is the
   trapezoidal integral of total power over the **decode window only** - model
   load and prefill are excluded from the output cost and reported separately.
   The decode window is anchored at `cmd_end - TAIL_S` (default 1.0 s) using
   the durations llama.cpp prints to stderr.
3. **Two figures per run**: total energy per token, and marginal energy
   (total minus `idle_power x duration`). Both land in `energy.json`.
4. **Discipline**: fixed power mode, fan at a fixed speed, headless, one
   variable at a time, median of 5 runs, idle measured on every run as a drift
   check (it stayed at 11.16-11.26 W across all 30 MAXN runs).

## Run directory layout

Each `runs/<date>-energy-<mode>-<variant>-<nn>/` holds:

| File | Content |
| --- | --- |
| `env.json` | Full board snapshot (JetPack, clocks, power mode, models) |
| `command.txt` | The exact command line |
| `stdout.log` / `stderr.log` | Generation output and llama.cpp statistics |
| `exit-code.txt` | Exit status (all protocol runs are 0) |
| `power.csv` | Raw samples: epoch, per-rail mW, total, VDDQ (unused) |
| `markers.txt` | Phase boundaries (idle start, command start/end, idle end) |
| `energy.json` | Computed result for that run |

`2026-08-12-energy-SMOKE-maxn-spec` is a validation run (5 s idle, 64 tokens),
kept for transparency and **excluded** from every aggregate.

## Session log

1. Smoke test first, out of protocol, to validate the whole chain end to end.
2. First `longctx` attempt aborted on the batch assert; fixed with `-b 20480`,
   failed run deleted, block restarted.
3. First long prompt generated 21,181 tokens, above the 20,480 context.
   Regenerated at 600 log lines (seed 42) for 15,718 tokens.
4. Block order: MAXN (q4km, iq4xs, spec, longctx, lcq4km, lciq4xs), reboot,
   30 W (q4km, lcq4km), reboot, back to MAXN.
5. No protocol run failed; all exit 0.
