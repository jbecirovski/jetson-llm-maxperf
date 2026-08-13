# Campaign 2026-08-12 - Speculative decoding, task by task

**Question**: the +59% measured in the previous campaign, is it a property of
the technique or of the task it was measured on?

51 runs: 5 task families x 5 prompts x (baseline + speculative), plus a drift
check. Prompts are sampled from the same public corpora the published
benchmarks use, and committed in `tasks/` with attribution.

## Results

| Task | Draft acceptance | tok/s | Gain vs paired baseline |
| --- | --- | --- | --- |
| Tool calls (strict schema) | 87% | 61.5 | x2.84 |
| Python code | 82% | 60.2 | x2.75 |
| JSON extraction | 66% | 51.1 | x2.39 |
| Summarization | 45% | 37.4 | x1.75 |
| Open-ended writing | 45% | 38.6 | x1.75 |

Paired baseline: 21.7 tok/s (same binary, drafting disabled).

Findings:

- The gain tracks **output predictability**: the more constrained the text,
  the better the draft guesses. Tool calls and code sit at the top.
- **Nothing collapsed.** The hypothesis that open-ended writing would fall
  below baseline was wrong: even the worst case gains 75%.
- **Output is not always identical.** Across 25 diffed pairs at temperature 0,
  11 diverge mid-text, all of them prose. Code and tool calls: 10/10
  identical. Hypothesis (labelled as such): batched validation shifts
  floating-point rounding, and near-ties between two tokens flip.

## Scripts

| Script | Usage |
| --- | --- |
| `scripts/benchmark-speculative-tasks.sh` | `./benchmark-speculative-tasks.sh` - runs the whole campaign (~50 runs plus drift check). Run `sudo ../../bench/maximize-perf.sh` first. |

Overridable environment: `LLAMA_BIN`, `MODEL`, `DRAFT`.

Design notes:

- One variable: the task (the prompt file). Everything else frozen and
  identical to the 2026-08-05 reference runs.
- **Paired runs**: each prompt runs twice, baseline and speculative, from the
  same binary (`llama-speculative`, `--spec-draft-n-max 0` disables drafting).
  This gives a per-task reference and the output diff at temperature 0.
- Fixed run order, first sequence replayed at the end as a thermal drift check
  (-1.6% on this session, healthy).
- `llama-cli` of the pinned build is a chat REPL that ignores `-no-cnv`/`-f`
  for scripted completion, which is why the baseline uses `llama-speculative`
  with drafting off.

See `tasks/README.md` for corpus sources and prompt selection.
