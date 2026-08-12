#!/usr/bin/env bash
# Per-task speculative decoding campaign: paired runs (baseline llama-cli +
# speculative llama-speculative) over the prompts in tasks/, with full proof
# capture via capture-jetson-env.sh --wrap. See tasks/README.md for corpus
# sources and design.
#
# Design:
#   - one variable: the task (the prompt file). Everything else frozen and
#     identical to the 2026-08-05 reference runs (MAXN + jetson_clocks first!).
#   - paired runs per prompt: the baseline gives a per-task reference AND the
#     output diff (temp 0) that upgrades "identical output guaranteed by the
#     mechanism" to "verified on this bench".
#   - fixed run order, plus a replay of the first speculative run at the end
#     (drift check: if it diverges from its first pass, the session is suspect).
#
# Usage:
#   sudo ./maximize-perf.sh          # once, before the campaign
#   ./benchmark-speculative-tasks.sh # ~50 runs + drift check
#
# Overridable environment:
#   LLAMA_BIN  (default: $HOME/llama.cpp/build/bin)
#   MODEL      (default: $HOME/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)
#   DRAFT      (default: $HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf)
#
# Flag note: --spec-draft-n-max matches the repo's pinned build (a035a88).
# Newer llama.cpp builds renamed it --draft-max; adjust if yours differs.
#
# Baseline note: the baseline uses the SAME llama-speculative binary with
# --spec-draft-n-max 0 (drafting fully disabled, n_drafted=0 in the stats).
# Same command, one parameter changes - the cleanest one-variable design.
# (llama-cli of build a035a88 is a chat REPL that ignores -no-cnv/-f for
# scripted completion and loops forever on stdin EOF - do not use it here.)

set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin}"
MODEL="${MODEL:-$HOME/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
DRAFT="${DRAFT:-$HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"

N_PREDICT=256
CTX=4096
DRAFT_MAX=8
COOLDOWN=30          # seconds between runs, thermal settling
DATE_TAG=$(date +%F)
OUT_DIR="benchmarks"
TASKS="toolcalls json-extract code summarize writing"

run_one() {  # $1 task  $2 prompt file  $3 variant: base|spec  $4 optional suffix
    local task=$1 pf=$2 variant=$3 suffix=${4:-}
    local nn dest latest
    nn=$(basename "$pf" .txt)
    dest="$OUT_DIR/${DATE_TAG}-task-${task}-${nn}-${variant}${suffix}"
    if [ -d "$dest" ]; then
        echo "skip (exists): $dest"
        return 0
    fi
    echo "=== $dest ==="
    local nmax=$DRAFT_MAX
    [ "$variant" = "base" ] && nmax=0
    ./capture-jetson-env.sh --wrap -- "$LLAMA_BIN/llama-speculative" \
        -m "$MODEL" -md "$DRAFT" -f "$pf" \
        -n $N_PREDICT -c $CTX -ngl 99 -ngld 99 \
        --temp 0 --spec-draft-n-max $nmax
    latest=$(ls -dt bench-run-* | head -1)
    mkdir -p "$OUT_DIR"
    mv "$latest" "$dest"
    sleep $COOLDOWN
}

for task in $TASKS; do
    for pf in tasks/$task/[0-9][0-9].txt; do
        run_one "$task" "$pf" base
        run_one "$task" "$pf" spec
    done
done

# Drift check: replay the very first speculative run under a distinct name.
run_one "toolcalls" "tasks/toolcalls/01.txt" spec "-driftcheck"

echo ""
echo "Campaign done. Quick extraction (both variants share the llama-speculative stderr format):"
echo "  spec stats     : grep -H -E 'decoded|n_drafted|n_accept|accept' $OUT_DIR/${DATE_TAG}-task-*-spec*/stderr.log"
echo "  baseline tok/s : grep -H 'decoded' $OUT_DIR/${DATE_TAG}-task-*-base/stderr.log"
echo "  output diffs   : for d in $OUT_DIR/${DATE_TAG}-task-*-base; do diff \"\$d/stdout.log\" \"\${d%-base}-spec/stdout.log\" >/dev/null && echo \"IDENTICAL \$d\" || echo \"DIFFERS   \$d\"; done"
