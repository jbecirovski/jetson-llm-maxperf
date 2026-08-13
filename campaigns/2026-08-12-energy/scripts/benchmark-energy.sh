#!/usr/bin/env bash
# Energy-per-token campaign runner (post #5). One variable per block: the
# config (model/draft variant here, the nvpmodel power mode is set manually
# and rebooted between blocks - it is captured in env.json by the wrapper).
#
# One run = INA3221 sampling at 10 Hz over [idle 60 s | generation | idle 60 s],
# generation under capture-jetson-env.sh --wrap (same proof convention as the
# task campaign), then compute-energy.py writes energy.json into the run dir.
#
# Baseline design carried over from the task campaign: every variant uses the
# SAME llama-speculative binary with the draft model loaded; only
# --spec-draft-n-max changes (0 = drafting disabled, n_drafted=0). One
# variable, constant referential. (llama-cli of build a035a88 is a chat REPL
# unusable in scripts - do not switch back.)
#
# Usage:
#   sudo ./maximize-perf.sh                # for MAXN blocks only
#   ./benchmark-energy.sh q4km   [runs=5]  # 8B Q4_K_M, draft disabled
#   ./benchmark-energy.sh iq4xs  [runs=5]  # 8B IQ4_XS, draft disabled
#   ./benchmark-energy.sh spec   [runs=5]  # 8B Q4_K_M + 1B draft, n-max 8
#   ./benchmark-energy.sh longctx [runs=5] # spec config, ~16K-token log in,
#                                          # small JSON out - measures the
#                                          # prefill (input) coefficient
#                                          # (wh_per_mtok_input in energy.json)
#   ./benchmark-energy.sh lcq4km  [runs=5] # same long-context run, q4km
#   ./benchmark-energy.sh lciq4xs [runs=5] # same long-context run, IQ4_XS
#
# Campaign sequence (reboot + nvpmodel between blocks, fixed order):
#   MAXN:  q4km, iq4xs, spec, longctx    then 30W mode: q4km (eco-mode trap)
#
# Env overrides: IDLE_SECS=60 COOLDOWN=30 N_PREDICT=256 CTX=4096 LLAMA_BIN=...
set -euo pipefail
cd "$(dirname "$0")"

VARIANT=${1:?variant: q4km | iq4xs | spec}
RUNS=${2:-5}
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin}"
M_Q4KM="$HOME/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
M_IQ4XS="$HOME/models/Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf"
DRAFT="$HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
IDLE_SECS=${IDLE_SECS:-60}
COOLDOWN=${COOLDOWN:-30}
N_PREDICT=${N_PREDICT:-256}
CTX=${CTX:-4096}
BATCH=${BATCH:-2048}
PROMPT=energy-prompt.txt
DATE_TAG=$(date +%F)

case "$VARIANT" in
  q4km)  MODEL=$M_Q4KM;  NMAX=0 ;;
  iq4xs) MODEL=$M_IQ4XS; NMAX=0 ;;
  spec)  MODEL=$M_Q4KM;  NMAX=8 ;;
  longctx)
    # robot use case: long context in, short structured output. Same config
    # as spec; the context needs a larger KV cache. The prompt file is
    # generated deterministically by make-longctx-prompt.py (seed 42).
    # -b must cover the whole prompt: llama-speculative submits it as ONE
    # llama_decode call (GGML_ASSERT n_tokens_all <= n_batch otherwise).
    MODEL=$M_Q4KM; NMAX=8; PROMPT=longctx-prompt.txt
    CTX=20480; N_PREDICT=192; BATCH=20480 ;;
  lcq4km)   # long-context prefill on the q4km baseline (input coefficient)
    MODEL=$M_Q4KM; NMAX=0; PROMPT=longctx-prompt.txt
    CTX=20480; N_PREDICT=192; BATCH=20480 ;;
  lciq4xs)  # long-context prefill on IQ4_XS (input coefficient)
    MODEL=$M_IQ4XS; NMAX=0; PROMPT=longctx-prompt.txt
    CTX=20480; N_PREDICT=192; BATCH=20480 ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

# power mode tag for the run folder name, e.g. maxn / 30w / 15w
MODE=$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: //p' \
       | tr '[:upper:]' '[:lower:]' | tr -d '_')
MODE=${MODE:-unknown}

for i in $(seq -w 1 "$RUNS"); do
  dest="benchmarks/${DATE_TAG}-energy-${MODE}-${VARIANT}-${i}"
  if [ -d "$dest" ]; then
    echo "skip (exists): $dest"
    continue
  fi
  echo "=== $dest ==="

  csv=$(mktemp /tmp/power-XXXX.csv)
  mrk=$(mktemp /tmp/markers-XXXX.txt)
  python3 power-sampler.py --out "$csv" --interval 0.1 &
  SAMPLER=$!
  trap 'kill $SAMPLER 2>/dev/null || true' EXIT
  sleep 2   # sampler warmup

  echo "idle_pre_start $(date +%s.%N)" >> "$mrk"
  sleep "$IDLE_SECS"
  echo "cmd_start $(date +%s.%N)" >> "$mrk"
  ./capture-jetson-env.sh --wrap -- "$LLAMA_BIN/llama-speculative" \
      -m "$MODEL" -md "$DRAFT" --spec-draft-n-max "$NMAX" \
      -c "$CTX" -b "$BATCH" -n "$N_PREDICT" -ngl 99 -ngld 99 --temp 0 \
      -f "$PROMPT"
  echo "cmd_end $(date +%s.%N)" >> "$mrk"
  sleep "$IDLE_SECS"
  echo "idle_post_end $(date +%s.%N)" >> "$mrk"

  kill "$SAMPLER" 2>/dev/null || true
  wait "$SAMPLER" 2>/dev/null || true
  trap - EXIT

  latest=$(ls -dt bench-run-* | head -1)
  mv "$latest" "$dest"
  mv "$csv" "$dest/power.csv"
  mv "$mrk" "$dest/markers.txt"
  python3 compute-energy.py "$dest"

  [ "$i" != "$(printf '%0*d' ${#i} "$RUNS")" ] && sleep "$COOLDOWN" || true
done

echo "block done: ${MODE}-${VARIANT} ($RUNS runs)"
