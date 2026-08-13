# tasks/ - per-task prompts for the speculative decoding campaign

Does the speculative decoding gain depend on the task? These prompts measure
it: 5 task families x 5 prompts, run paired (baseline + speculative) by
`../benchmark-speculative-tasks.sh`, same frozen config as the 2026-08-05
reference runs.

## Design borrowed from published benchmarks

Prompts are sampled from the same public corpora used by the speculative
decoding literature (Spec-Bench, ACL 2024 Findings; BFCL), so the resulting
numbers are directly comparable to published per-task grids - measured here on
edge hardware with llama.cpp's simple draft-model pipeline instead of server
GPUs with EAGLE/Medusa.

**Selection rule (deterministic)**: the first 5 entries of each source in its
canonical order, fetched on 2026-08-11. Each `NN.txt` has a `NN.source.txt`
next to it with the exact provenance (dataset, id, URL).

| Task dir | Corpus | License | Hypothesis to test |
| --- | --- | --- | --- |
| `toolcalls/` | BFCL v4 `parallel` category (ShishirPatil/gorilla) | Apache-2.0 | high acceptance |
| `json-extract/` | CNN/DailyMail 3.0.0 test articles (offsets 5-9) + a bench-specific extraction schema - this category is absent from published benchmarks | see CNN/DM terms | high acceptance |
| `code/` | HumanEval (openai/human-eval), field `prompt` | MIT | high acceptance |
| `summarize/` | CNN/DailyMail 3.0.0 test articles (offsets 0-4) | see CNN/DM terms | medium acceptance |
| `writing/` | MT-Bench `writing` category (lm-sys/FastChat) | Apache-2.0 | low acceptance, throughput possibly below the no-draft baseline |

## Prompt format

Each file is a raw completion prompt with the Llama 3.1 chat template applied
in-file (system + user + assistant header), **without** `<|begin_of_text|>`:
llama.cpp adds BOS itself.

Both runs of a pair use the **same `llama-speculative` binary** reading the
same file via `-f`; the only difference is `--spec-draft-n-max`: 0 for the
baseline (drafting fully disabled, `n_drafted=0` in the stats), 8 for the
measured run. The target model sees byte-identical input and code path in both
runs, which is what makes the output diff at temp 0 meaningful.

(Why not `llama-cli`: the chat-REPL rewrite in build a035a88 ignores
`-no-cnv`/`-f` for scripted completion and loops forever on stdin EOF.)

## Known caveats (document with the results, do not hide)

- **temp 0 everywhere, including creative writing**: the price of the output
  diff and of the one-variable rule. Real creative use runs hotter, which would
  degrade draft acceptance further - so the writing numbers here are a *floor*
  for the penalty, not a ceiling.
- **Output lengths differ by task**: toolcalls answers are short (a JSON array
  of calls), writing answers hit the `-n 256` cap. Decode tok/s is still
  comparable (per-token rate), but report generated-token counts alongside.
- **CNN/DailyMail articles are truncated to 5000 characters** to stay well
  inside `-c 4096` after templating.
- MT-Bench writing was chosen over chat-style prompts on purpose: RLHF-register
  chat is lexically predictable and can show surprisingly *high* acceptance
  (arXiv 2604.14682) - it would blur the "open-ended" end of the spectrum.

## Metrics to extract per run

| Metric | Where |
| --- | --- |
| Draft acceptance (n_accept / n_drafted) | `stderr.log` of the spec run |
| Decode tok/s (spec) | `stderr.log` of the spec run |
| Decode tok/s (baseline, same prompt) | `stderr.log` of the base run (eval time) |
| Mean accepted length per target read | derived: n_predict / (n_predict - n_accept) |
| Output identical? | `diff` of the two `stdout.log` (temp 0) |

Primary reported metric: per-task speedup vs the *same task's* baseline
(Spec-Bench convention), plus acceptance rate and mean accepted length
(SPEED-Bench convention).
