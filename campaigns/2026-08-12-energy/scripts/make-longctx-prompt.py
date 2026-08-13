#!/usr/bin/env python3
"""Generate the long-context prompt for the 'longctx' energy variant.

Deterministic (fixed seed): ~16K tokens of synthetic mobile-robot inspection
logs, ending with an instruction to output ONLY a small fixed-schema JSON.
This measures the robot use case 'long context in, short structured output':
the prefill (input) energy coefficient dominates and is reported by
compute-energy.py as wh_per_mtok_input.

Usage: python3 make-longctx-prompt.py > longctx-prompt.txt
The generated file is committed as-is; this script documents its provenance.
"""
import random

random.seed(42)

ZONES = ["dock-A", "dock-B", "aisle-1", "aisle-2", "aisle-3", "ramp-N",
         "ramp-S", "charge-bay", "cold-room", "mezzanine"]
EVENTS = [
    ("INFO", "waypoint reached, localization confidence {c:.2f}"),
    ("INFO", "lidar scan ok, {n} points, {o} dynamic obstacles tracked"),
    ("INFO", "battery at {b}%, cell delta {d} mV, temp {t:.1f} C"),
    ("INFO", "inspection photo captured, blur score {c:.2f}"),
    ("WARN", "path replanned, clearance {m:.2f} m below threshold"),
    ("WARN", "wifi rssi {r} dBm, telemetry buffered {k} messages"),
    ("WARN", "wheel slip detected, odometry drift {m:.2f} m corrected"),
    ("ERROR", "camera frame dropped, exposure retry {k}"),
    ("ERROR", "gauge unreadable at {z}, glare suspected, flagged for review"),
]

lines = []
t = 0.0
for i in range(600):  # ~16K tokens measured on the bench tokenizer (Llama 3.1)
    t += random.uniform(0.4, 3.2)
    lvl, tpl = random.choices(EVENTS, weights=[30, 25, 15, 10, 6, 5, 4, 3, 2])[0]
    msg = tpl.format(
        c=random.uniform(0.55, 0.99), n=random.randint(9000, 42000),
        o=random.randint(0, 6), b=random.randint(18, 100),
        d=random.randint(2, 40), t=random.uniform(21.0, 44.0),
        m=random.uniform(0.05, 0.9), r=random.randint(-88, -40),
        k=random.randint(1, 120), z=random.choice(ZONES),
    )
    lines.append(f"[{t:9.1f}] {lvl:5s} {random.choice(ZONES):10s} {msg}")

print("You are the reporting module of a warehouse inspection robot. "
      "Below is the raw mission log. Read it fully, then answer.\n")
print("\n".join(lines))
print("\nOutput ONLY a JSON object, no prose, with exactly these fields: "
      '{"total_entries": int, "errors": int, "warnings": int, '
      '"battery_min_pct": int, "zones_visited": int, '
      '"anomalies": [up to 5 short strings]}')
