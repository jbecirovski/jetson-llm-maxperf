#!/usr/bin/env bash
# capture-jetson-env.sh - snapshot the full benchmark-relevant configuration
# of an NVIDIA Jetson board as JSON, so results can be compared apples-to-apples.
#
# Usage:
#   ./capture-jetson-env.sh                      # print JSON to stdout
#   ./capture-jetson-env.sh -o env.json          # write JSON to file
#   ./capture-jetson-env.sh --wrap -- <command>  # run a benchmark command and
#                                                # save env + command + exit code
#                                                # + stdout/stderr in a run dir
set -u

OUT=""
WRAP=0
while [ $# -gt 0 ]; do
  case "$1" in
    -o) [ $# -ge 2 ] || { echo "-o requires a file path" >&2; exit 2; }
        OUT="$2"; shift 2 ;;
    --wrap) WRAP=1; shift; [ "${1:-}" = "--" ] && shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

json_escape() {
  printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null \
    || printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr -d '\n')"
}

read_or_null() {
  if [ -r "$1" ]; then
    json_escape "$(tr -d '\0' < "$1")"
  else
    printf 'null'
  fi
}

cmd_or_null() {
  local out
  if out=$("$@" 2>/dev/null) && [ -n "$out" ]; then
    json_escape "$out"
  else
    printf 'null'
  fi
}

num_or_null() {
  local out
  out=$("$@" 2>/dev/null)
  case "$out" in
    ''|*[!0-9]*) printf 'null' ;;
    *) printf '%s' "$out" ;;
  esac
}

capture() {
  local board_model l4t jetpack cuda_version nvpmodel_q jclocks mem_kb swap_kb kernel distro in_container

  board_model=$(read_or_null /proc/device-tree/model)
  l4t=$(read_or_null /etc/nv_tegra_release)
  jetpack=$(cmd_or_null sh -c "dpkg-query --show --showformat='\${Version}' nvidia-jetpack")
  cuda_version=$(read_or_null /usr/local/cuda/version.json)
  [ "$cuda_version" = "null" ] && cuda_version=$(cmd_or_null sh -c "nvcc --version | tail -n2")
  nvpmodel_q=$(cmd_or_null sudo -n nvpmodel -q)
  [ "$nvpmodel_q" = "null" ] && nvpmodel_q=$(cmd_or_null nvpmodel -q)
  jclocks=$(cmd_or_null sudo -n jetson_clocks --show)
  [ "$jclocks" = "null" ] && jclocks=$(cmd_or_null jetson_clocks --show)
  mem_kb=$(num_or_null sh -c "awk '/MemTotal/{print \$2}' /proc/meminfo")
  swap_kb=$(num_or_null sh -c "awk '/SwapTotal/{print \$2}' /proc/meminfo")
  kernel=$(cmd_or_null uname -r)
  distro=$(cmd_or_null sh -c ". /etc/os-release && echo \"\$PRETTY_NAME\"")
  if [ -f /.dockerenv ]; then in_container=true; else in_container=false; fi

  cat <<EOF
{
  "captured_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "board": {
    "model": $board_model,
    "mem_total_kb": $mem_kb,
    "swap_total_kb": $swap_kb
  },
  "software": {
    "l4t_release": $l4t,
    "jetpack_package": $jetpack,
    "cuda": $cuda_version,
    "kernel": $kernel,
    "distro": $distro,
    "in_container": $in_container
  },
  "power": {
    "nvpmodel_query": $nvpmodel_q,
    "jetson_clocks_show": $jclocks
  }
}
EOF
}

if [ "$WRAP" -eq 1 ]; then
  [ $# -gt 0 ] || { echo "--wrap requires a command after --" >&2; exit 2; }
  RUN_DIR="bench-run-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RUN_DIR"
  capture > "$RUN_DIR/env.json"
  printf '%q ' "$@" > "$RUN_DIR/command.txt"
  printf '\n' >> "$RUN_DIR/command.txt"
  "$@" > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)
  status=$?
  wait
  printf '%s\n' "$status" > "$RUN_DIR/exit-code.txt"
  echo "run saved in $RUN_DIR (env.json, command.txt, stdout.log, stderr.log, exit-code.txt)" >&2
  exit "$status"
elif [ -n "$OUT" ]; then
  capture > "$OUT"
  echo "environment written to $OUT" >&2
else
  capture
fi
