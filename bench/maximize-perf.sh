#!/usr/bin/env bash
# maximize-perf.sh - put a Jetson board in its maximum-throughput configuration
# for LLM inference: highest power mode (MAXN) + locked max clocks.
#
# Order matters: nvpmodel MUST run before jetson_clocks. Per NVIDIA's Developer
# Guide, once jetson_clocks has locked the clocks, changing the power mode
# requires a reboot.
#
# Usage:
#   sudo ./maximize-perf.sh          # set MAXN + lock clocks
#   sudo ./maximize-perf.sh --fan    # same, plus max fan speed
set -eu

FAN=""
[ "${1:-}" = "--fan" ] && FAN="--fan"

if [ "$(id -u)" -ne 0 ]; then
  echo "must run as root: sudo $0 $*" >&2
  exit 1
fi

command -v nvpmodel >/dev/null || { echo "nvpmodel not found - is this a Jetson?" >&2; exit 1; }

echo "current power mode:"
nvpmodel -q

MAXN_ID=$(awk -F'[= >]' '/^< *POWER_MODEL/ { id=""; name="";
  for (i=1; i<=NF; i++) { if ($i=="ID") id=$(i+1); if ($i=="NAME") name=$(i+1) }
  if (name ~ /^MAXN/) { print id; exit } }' /etc/nvpmodel.conf)

if [ -z "$MAXN_ID" ]; then
  echo "no MAXN mode found in /etc/nvpmodel.conf - set the highest mode manually with: nvpmodel -m <id>" >&2
  exit 1
fi

current_mode() {
  nvpmodel -q 2>/dev/null | awk -F': ' '/NV Power Mode/ {print $2; exit}'
}

case "$(current_mode)" in
  MAXN*)
    echo "already in MAXN mode, skipping nvpmodel."
    ;;
  *)
    echo "switching to MAXN (mode $MAXN_ID)..."
    if [ -t 0 ]; then
      nvpmodel -m "$MAXN_ID"
    else
      # nvpmodel can prompt for confirmation (reboot, online-core changes);
      # without a terminal, feed EOF instead of hanging forever
      nvpmodel -m "$MAXN_ID" < /dev/null || true
    fi
    case "$(current_mode)" in
      MAXN*) ;;
      *)
        echo "power mode is still not MAXN - nvpmodel needs interactive confirmation or a reboot. Rerun from a terminal, then run this script again." >&2
        exit 1
        ;;
    esac
    ;;
esac

echo "locking clocks to maximum..."
jetson_clocks $FAN

echo "done. current state:"
nvpmodel -q
jetson_clocks --show

echo
echo "note: to change power mode again, reboot first (clocks are now locked)."
