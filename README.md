# jetson-llm-maxperf

Get the maximum LLM tokens/sec out of an NVIDIA Jetson board, and prove it.

On one AGX Orin 64GB, the same 8B model generates anywhere between 8.1 and
43.0 tok/s in our own measurements - a 5.3x gap driven entirely by power
mode, quantization and decoding strategy (the JetPack version moves it by
1%). Published figures spread even wider, from 1.8 to a claimed 52 tok/s
(see sources below). This repo gives you, in order:

1. a script that puts the board in its maximum-performance configuration,
2. the two install paths for a fast LLM runtime (containers or native build),
3. a script that captures the full configuration next to your results, so your
   numbers are comparable and reproducible,
4. a per-task speculative decoding test bench that measures how much a draft
   model buys you on YOUR workload - and whether the output really stays
   identical,
5. an energy test bench that turns those tok/s into Wh per million tokens,
   the unit that matters once the board runs on a battery.

![3D bar chart of this repo's measured generation throughput on AGX Orin 64GB, all with Llama 3.1 8B: 8.1 tok/s in 30W power mode with stock clocks, 26.8 in MAXN with locked clocks under JetPack 6.0, 27.1 under JetPack 6.2.1, 34.5 with the smaller IQ4_XS quant, and 43.0 with speculative decoding using a 1B draft - the only full-green bar escaping the shaded zone of the 41.6 tok/s single-stream bandwidth ceiling, marked under the axis](assets/runtime-29x.png)

Target platform: AGX Orin under JetPack 6.2 (L4T r36.4.x). Figures below are
from the sources at the bottom.

## Repository map

```text
bench/                  the common harness: environment capture, max-perf script
campaigns/              one directory per measurement campaign
  2026-08-05-runtime-power/       runtime, quantization, power modes (posts 1-2)
  2026-08-12-speculative-tasks/   speculative gain per task family (post 4)
  2026-08-12-energy/              energy per token, input and output (post 5)
assets/                 charts published with the campaigns
```

Every campaign directory holds its own `README.md` (question, results,
scripts, how to reproduce), its `scripts/`, and one proof directory per run
under `runs/`. Start with `bench/README.md` if you want the harness, or with a
campaign README if you want the numbers.

| Campaign | Question | Runs |
| --- | --- | --- |
| [2026-08-05 runtime & power](campaigns/2026-08-05-runtime-power/) | What moves throughput, and where is the ceiling? | 26 |
| [2026-08-12 speculative tasks](campaigns/2026-08-12-speculative-tasks/) | Is the speculative gain a property of the task? | 51 |
| [2026-08-12 energy](campaigns/2026-08-12-energy/) | What does a million tokens cost on battery? | 41 |

## Step 1 - Put the board in max-perf mode

```bash
sudo ./bench/maximize-perf.sh        # MAXN power mode + locked max clocks
sudo ./bench/maximize-perf.sh --fan  # same, plus max fan speed
```

What it does, in the order required by NVIDIA's docs:

1. `nvpmodel -m <MAXN id>` - switches to the highest power mode (the id is
   read from `/etc/nvpmodel.conf`, it varies across boards). Measured on our
   board: **3.3x generation throughput** between 30W and MAXN (see Measured
   results). Note: switching between modes with different online-core counts
   requires a reboot, and nvpmodel prompts for it - the script fails cleanly
   instead of hanging when run non-interactively.
2. `jetson_clocks` - locks CPU/GPU/EMC clocks to their maximum. Must run
   *after* nvpmodel: once clocks are locked, changing the power mode requires
   a reboot. Measured effect on sustained throughput: ~+1% - its real value
   is measurement stability (see Measured results).

## Step 2 - Install a fast runtime

Two paths; both give comparable results.

### Path A - jetson-containers (no build, recommended to start)

```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
jetson-containers run $(autotag llama_cpp)   # or: ollama
```

`autotag` picks the container image matching your JetPack/L4T version, which
removes the main source of slow setups (a runtime built without CUDA or for
the wrong JetPack).

### Path B - native llama.cpp CUDA build

Test bench: [llama.cpp](https://github.com/ggml-org/llama.cpp) (its
`llama-bench` binary is the benchmark tool used in Step 3). The build flags
below match the published 52 tok/s claim (Llama 3.1 8B, Q4_K_M, AGX Orin
64GB, JetPack 6.2) - a figure our own measurements could not reproduce and
that exceeds the memory-bandwidth ceiling (see Measured results):

The commands below are the full sequence validated on our test board (see
Measured results), starting from a stock JetPack install.

```bash
# 1. Prerequisites (cmake is NOT preinstalled on stock JetPack; nvcc exists
#    but is not on PATH)
sudo apt-get update && sudo apt-get install -y cmake git
export PATH=/usr/local/cuda/bin:$PATH

# 2. Get and build llama.cpp (CUDA build, ~15-20 min on the 12 cores)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="87" \
  -DGGML_CUDA_F16=ON \
  -DLLAMA_CURL=OFF
cmake --build build -j12 --target llama-bench llama-cli

# 3. Download a model (public GGUF, ~4.9 GB, no HuggingFace token needed)
mkdir -p ~/models
wget -c -O ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# 4. Run
./build/bin/llama-cli -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 99 -c 4096
```

The flags that matter (all hardware- or build-related - nothing here is
specific to one model; the same binary runs any GGUF file):

- `-DGGML_CUDA=ON` - without it you get the CPU-only build (the 1.8 tok/s
  scenario).
- `-DCMAKE_CUDA_ARCHITECTURES="87"` - compiles the CUDA kernels for Orin's
  compute capability.
- `-DLLAMA_CURL=OFF` - avoids a build failure when libcurl-dev is not
  installed (it is not, on a stock JetPack).
- `--target llama-bench llama-cli` - builds only the two useful binaries,
  much faster than the full project.
- `-ngl 99` - offload all layers to GPU (runtime flag, not build).
- Q4_K_M quantization - the practical sweet spot on Jetson (quality vs
  memory footprint); IQ4_XS of the same model is ~25% faster at slightly
  lower quality (see Measured results).

## Step 3 - Benchmark with proof

A throughput number without its configuration is not comparable. Capture the
environment next to every run:

```bash
# Environment snapshot only
./bench/capture-jetson-env.sh -o env.json

# Wrap a benchmark: saves env + command + outputs + exit code
# (paths follow Path B above: llama.cpp cloned in ~, model in ~/models)
./bench/capture-jetson-env.sh --wrap -- ~/llama.cpp/build/bin/llama-bench \
  -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 99
```

`--wrap` creates a `bench-run-<timestamp>/` directory containing `env.json`,
`command.txt`, `stdout.log`, `stderr.log`, `exit-code.txt`. That full
directory is what makes a number citable.

Captured fields:

| Field | Source |
| --- | --- |
| Board model (e.g. AGX Orin 64GB) | `/proc/device-tree/model` |
| Total RAM and swap | `/proc/meminfo` |
| L4T version | `/etc/nv_tegra_release` |
| JetPack version | `nvidia-jetpack` package |
| CUDA version | `/usr/local/cuda/version.json` or `nvcc` |
| Power mode (MAXN, 30W...) | `nvpmodel -q` |
| Clock state | `jetson_clocks --show` |
| Kernel, distro, running in container | `uname`, `os-release`, `/.dockerenv` |

Any missing field (command unavailable, no sudo rights) is reported as `null`
instead of failing the capture.

## Step 4 - Measure how the speculative gain depends on YOUR task

`campaigns/2026-08-12-speculative-tasks/scripts/benchmark-speculative-tasks.sh` is the test bench behind a question the
speculative decoding result leaves open: is the +59% a property of the
technique, or of the task it ran on? It answers by running a fixed campaign
of 51 wrapped runs (~35 min unattended):

- **5 task families x 5 prompts**, stored in `campaigns/2026-08-12-speculative-tasks/tasks/` and sampled from the
  same public corpora the speculative decoding literature benchmarks on:
  tool calls (BFCL v4), Python code (HumanEval), JSON extraction and
  summarization (CNN/DailyMail), open-ended writing (MT-Bench). Corpora,
  licenses, sampling rule and prompt format: [tasks/README.md](campaigns/2026-08-12-speculative-tasks/tasks/README.md).
- **Paired runs, one variable.** For every prompt, the baseline and the
  speculative run use the *same* `llama-speculative` binary with the same
  flags; only `--spec-draft-n-max` changes (0 = drafting fully disabled,
  8 = the crest found above). Same code path, same overhead - so the
  per-task speedups compare like with like. This paired baseline (~21.7
  tok/s) is NOT the llama-bench tg128 (27.1): different instrument, do not
  mix them.
- **Proof per run**: every run goes through the Step 3 wrapper and lands in
  `campaigns/2026-08-12-speculative-tasks/runs/<date>-task-<family>-<nn>-{base,spec}/`. Existing directories
  are skipped, so an interrupted campaign resumes where it stopped.
- **Thermal drift check**: the first speculative run is replayed at the end
  (`-driftcheck` suffix). If it moved, the session throttled and the numbers
  are suspect.

```bash
# Prerequisites: Step 1 applied, Path B build plus the speculative example:
cmake --build build -j12 --target llama-speculative

# The 1B draft model next to the 8B target (public GGUF, ~0.8 GB):
wget -c -O ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Run the campaign (from the repo root, next to capture-jetson-env.sh)
./campaigns/2026-08-12-speculative-tasks/scripts/benchmark-speculative-tasks.sh

# Paths are overridable if your layout differs:
LLAMA_BIN=~/llama.cpp/build/bin MODEL=~/models/... DRAFT=~/models/... \
  ./campaigns/2026-08-12-speculative-tasks/scripts/benchmark-speculative-tasks.sh
```

On completion the script prints one-liners to extract acceptance rates,
tok/s and output diffs from the proof directories. Results from our board:
see "The gain is a property of the task" below.

## Measured results

All runs below share the same setup: AGX Orin 64GB Developer Kit, MAXN power
mode with locked clocks (Step 1), llama.cpp commit `a035a88` built per Path B,
target model Llama 3.1 8B Instruct. Every number links to its full proof
directory (env.json, command, logs, exit code) committed under `campaigns/*/runs/`,
produced with the Step 3 wrapper. `pp512` is prompt processing, `tg128` is
generation (the number that matters for an interactive agent).

### Baseline: llama-bench, single stream

| JetPack | Model file | pp512 tok/s | tg128 tok/s | Proof |
| --- | --- | --- | --- | --- |
| 6.0 (CUDA 12.2) | Q4_K_M (4.58 GiB) | 969.2 ± 12.7 | 26.8 ± 0.1 | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.0/) |
| 6.0 (CUDA 12.2) | IQ4_XS (4.13 GiB) | 1122.1 ± 30.3 | 33.5 ± 0.9 | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.0-iq4xs/) |
| 6.2.1 (CUDA 12.6) | Q4_K_M (4.58 GiB) | 1001.7 ± 8.7 | 27.1 ± 0.0 | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.2.1/) |
| 6.2.1 (CUDA 12.6) | IQ4_XS (4.13 GiB) | 1195.0 ± 12.7 | 34.5 ± 0.1 | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.2.1-iq4xs/) |

### What Step 1 actually buys you, decomposed

Same board, same JetPack 6.2.1, same build, same Q4_K_M model - only the
power settings change (each step measured after a fresh reboot):

| Power configuration | pp512 tok/s | tg128 tok/s | Proof |
| --- | --- | --- | --- |
| 30W mode, stock clocks | 249.6 ± 0.7 | 8.1 ± 0.1 | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.2.1-30w-stock/) |
| MAXN mode, stock clocks | 992.4 ± 10.9 | 26.9 ± 0.2 | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.2.1-maxn-stock-clocks/) |
| MAXN + `jetson_clocks` | 1002.2 ± 8.8 | 27.1 ± 0.0 | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.2.1-maxn-locked-clocks/) |

Two findings worth more than the folklore:

- **The power mode is the whole story: 3.3x on generation** (8.1 → 26.9),
  4x on prefill. This is on the *same* 64GB board - bigger than the 2.5x from
  the GitHub thread, which mixed power mode with a board-variant change.
- **Locking clocks barely moves sustained throughput (+0.7%)** - under
  continuous load, DVFS ramps the clocks up by itself. What `jetson_clocks`
  actually buys is *stability*: the tg128 standard deviation drops from ±0.16
  to ±0.01. Lock the clocks so your numbers are reproducible, not to make
  them bigger.

### Why generation is stuck around 30 tok/s: the bandwidth ceiling

Generating one token requires reading the *entire* model weights from RAM -
the on-chip caches are MB-scale, the model is GB-scale. So single-stream
generation is bounded by:

```text
max tok/s = memory bandwidth / model size
          = 204.8 GB/s / 4.92 GB (Q4_K_M)  ≈ 41.6 tok/s
          = 204.8 GB/s / 4.43 GB (IQ4_XS)  ≈ 46.2 tok/s
```

Our measurements sit at 65-75% of those ceilings, which is normal llama.cpp
efficiency. This one formula explains most of the table:

- **Smaller file, faster generation**: IQ4_XS is the same model with ~10%
  fewer bytes to read per token, and generates ~27% faster.
- **JetPack version is irrelevant here**: upgrading 6.0 → 6.2.1 (CUDA
  12.2 → 12.6, rebuilt) moved generation by +1%. Software does not add
  memory bandwidth. The runtime *build* (CPU-only vs CUDA) makes orders of
  magnitude; the JetPack *version* does not.
- **Prefill is 35x faster than generation** on the same hardware because it
  amortizes each weight read over 512 tokens - it is compute-bound, not
  bandwidth-bound.

### Fact-checking published figures with that formula

- NVIDIA's 47 tok/s ([jetson-containers MLC
  README](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/mlc)):
  Llama-2-**7B** (~3.6 GB) under MLC `q4f16_ft` kernels → ceiling ~57 tok/s,
  measured 46.9 = ~82% efficiency. **Plausible** - smaller model, more
  efficient runtime.
- The 52 tok/s claim for an 8B Q4_K_M under llama.cpp: **exceeds the 41.6
  physical ceiling** and should not be treated as reproducible.

### Beating the ceiling without touching quantization: speculative decoding

A small draft model (Llama 3.2 1B, 0.8 GB) proposes several tokens; the 8B
verifies the whole batch in a *single* weight pass. The "one token = one
full read" rule no longer binds, and by construction the 8B only keeps
tokens it would have picked itself - identical output in exact arithmetic
(see the per-task campaign below for what batched floating-point does to
that guarantee in practice). Measured on a technical prompt (JetPack 6.2.1,
Q4_K_M target, `llama-speculative`):

| Draft length (`--spec-draft-n-max`) | Acceptance | Decode tok/s | Proof |
| --- | --- | --- | --- |
| none (baseline tg128) | - | 27.1 | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.2.1/) |
| 3 (default) | 69.9% | 32.9 | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.2.1-speculative/) |
| **8** | 53.8% | **43.0** | [dir](campaigns/2026-08-05-runtime-power/runs/2026-08-05-agx-orin-64gb-jetpack-6.2.1-speculative-n8/) |
| 16 | 27.8% | 28.2 | (raw run) |

**+59% over the baseline, above the naive single-stream ceiling.** Two honest
caveats: draft length is a crest, not a slope - drafting 16 tokens collapses
acceptance and destroys the gain - and the speedup depends on how predictable
the generated text is. That second caveat stopped being folklore on
2026-08-12: it is measured properly below.

### The gain is a property of the task (campaign of 2026-08-12)

Step 4 campaign: same frozen setup, `--spec-draft-n-max 8`, temp 0, `-n 256`,
5 prompts per task family, paired against the same binary with drafting
disabled. Means over the 5 prompts; proof directories under
`campaigns/2026-08-12-speculative-tasks/runs/` (51 runs, all exit 0, drift check -1.6%).

![3D bar chart of measured generation throughput on NVIDIA Jetson AGX Orin 64GB, Llama 3.1 8B Q4_K_M with a 1B draft under llama.cpp, temp 0, mean of 5 prompts per task. Bottom gray bar: 21.7 tokens per second without draft, same binary. Light green bars below the 41.6 bandwidth ceiling: document summarization 37.4 (x1.75) and open-ended writing 38.6 (x1.75). Full green bars above the ceiling: JSON extraction 51.1 (x2.39), Python code generation 60.2 (x2.75), tool calls on a strict schema 61.5 (x2.84). The area below the 41.6 tok/s ceiling is shaded light gray with a marker under the axis, draft acceptance rate under each task label.](assets/speculative-by-task.png)

| Task family (corpus) | Acceptance | Base tok/s | Spec tok/s | Speedup | Identical output |
| --- | --- | --- | --- | --- | --- |
| Tool calls, strict schema (BFCL v4) | 87.1% | 21.6 | 61.5 | **2.84x** | 5/5 |
| Python code generation (HumanEval) | 81.6% | 21.9 | 60.2 | 2.75x | 5/5 |
| JSON extraction (CNN/DailyMail) | 65.8% | 21.4 | 51.1 | 2.39x | 3/5 |
| Document summarization (CNN/DailyMail) | 44.7% | 21.3 | 37.4 | 1.75x | 0/5 |
| Open-ended writing (MT-Bench) | 45.4% | 22.0 | 38.6 | 1.75x | 1/5 |

Three findings:

- **The gain tracks output predictability** - from 1.75x to 2.84x at strictly
  identical config. Constrained formats (tool calls, code, JSON) are exactly
  what an embedded agent generates, and they sit at the top.
- **Nothing collapsed at draft length 8**: even open-ended writing gains 75%.
  With a same-family 1B draft, the collapse lives in the draft length (16,
  table above), not in the task.
- **"Identical output" has fine print.** At temp 0, 11 of 25 pairs diverge
  mid-text - all in prose (summarization 5/5, writing 4/5, JSON extraction
  2/5); code and tool calls are 10/10 identical. Working hypothesis, not yet
  verified mechanically: batched validation changes floating-point rounding,
  and near-ties between candidate tokens flip. The diff tolerates one known
  artifact: the speculative loop overruns EOS by the rest of its final burst.

### What a token costs in energy (campaign of 2026-08-12)

Tokens per second is the wrong unit on a battery-powered robot. The
[energy campaign](campaigns/2026-08-12-energy/) measures the onboard power
rails at 10 Hz during generation and divides the integrated energy by the
tokens produced. Medians of 5 runs, short-context block:

![3D bar chart of the measured energy cost per million generated tokens on NVIDIA Jetson AGX Orin 64GB, Llama 3.1 8B under llama.cpp, onboard INA3221 rails, median of 5 runs per config. Gray bar, the 30 W efficiency mode: 754 Wh per million tokens at 6.6 tok/s and 18.0 W, annotated as the trap - fewer watts, more energy per token. Light green bars: MAXN 602 Wh at 22.0 tok/s and 47.6 W, MAXN with IQ4_XS 501 Wh at 26.8 tok/s and 48.3 W. Full green bar, the cheapest config: MAXN with a 1B draft 249 Wh at 46.8 tok/s and 42.0 W, annotated as 3x less than eco mode.](assets/energy-per-token.png)

| Config | tok/s | W decode | Wh per million tokens |
| --- | --- | --- | --- |
| 30 W "efficiency" mode | 6.6 | 18.0 | **754** |
| MAXN | 22.0 | 47.6 | **602** |
| MAXN + IQ4_XS | 26.8 | 48.3 | **501** |
| MAXN + 1B draft (speculative) | 46.8 | 42.0 | **249** |

What the campaign README details:

- **Race-to-idle wins.** The "efficiency" mode draws 2.6x fewer watts and
  costs **25% more energy per token**, because it stays busy 3.3x longer.
- **Speculative decoding draws less power, not more**, despite loading a
  second model: 42.0 W against 47.6 W at baseline, so 2.4x less energy per
  token. Datacenter literature often reports no energy gain from a plain
  draft model; this is an edge-specific result.
- **Reading context is nearly free and flat**: 16-18 Wh per million input
  tokens on every MAXN config, against 249-602 for output. Prefill saturates
  compute rather than memory bandwidth, so making generation faster does not
  make reading cheaper.
- **A loaded context makes every output token dearer**: with 16K tokens in
  the KV cache, generation halves at constant power, so the output
  coefficient rises by ~50% (249 to 372 Wh/Mtok on the speculative config,
  602 to 910 on the baseline).
- **The worst edge config exceeds the datacenter reference**: 30 W mode with
  a 16K context lands at 1025 Wh/Mtok, above the ~970 measured on an A100.
  Edge is not frugal by nature, the configuration decides.

All numbers are a lower bound: onboard rails measure downstream of the
regulators (see the campaign README for the published calibration).

## Known limitations

- **Validated on one board so far**: AGX Orin 64GB Developer Kit, under both
  JetPack 6.0 (L4T r36.3) and 6.2.1 (L4T r36.4.7) - environment capture,
  MAXN switch + clock locking (GPU verified at max frequency), and `--wrap`
  runs. Not yet validated on other board variants (Orin Nano/NX, 32GB).
- MAXN draws maximum power and heat; on battery-powered robots, pick the
  power mode from your power budget instead and record it with the capture
  script - comparable beats maximal in the field.
- The benchmark's own config (quantization, context, batch) is captured via
  `command.txt` in `--wrap` mode, not parsed.

## Sources

Benchmarks behind the 29x gap:

- Late-2023 benchmark (LLaMA2-7B, llama.cpp/GGML q2/q4, AGX Orin 64GB,
  1.3-1.8 tok/s): <https://www.dfrobot.com/blog-13496.html>
- 2026 benchmark (Llama 3.1 8B, llama.cpp CUDA build, Q4_K_M, JetPack 6.2,
  jetson_clocks, 52 tok/s on AGX Orin 64GB), including the build flags used
  in Path B: <https://proventusnova.com/blog/llm-inference-jetson-orin-llamacpp-ollama/>
- The 19 vs 47 tok/s GitHub thread (30W vs MAXN mode, 32 vs 64GB board):
  <https://github.com/dusty-nv/jetson-containers/issues/532>
- The 47 tok/s figure (Llama-2-7B, MLC INT4) comes from the Jetson AI Lab
  benchmarks page of the time, since archived; quoted as-is in the thread above.

Official documentation for the interfaces used by the scripts (Jetson Linux
r36.4.3, the JetPack 6.2 release):

- `nvpmodel` (query/set power modes, MAXN, power budgets) and `jetson_clocks`
  (`--show`, `--store`, `--restore`; section "Maximizing Jetson Orin Series
  Performance"), both on the same page:
  <https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html>
- `/etc/nv_tegra_release` (L4T version) and `/proc/device-tree/model` (board
  variant): standard L4T conventions with no dedicated doc page; used notably
  by jetson-containers for autodetection:
  <https://github.com/dusty-nv/jetson-containers>

Test bench and model used for validation:

- llama.cpp (runtime and `llama-bench` benchmark tool):
  <https://github.com/ggml-org/llama.cpp>
- Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (public GGUF quantization):
  <https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF>
