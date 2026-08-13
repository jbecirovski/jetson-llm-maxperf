# Campaign 2026-08-05 - Runtime, quantization and power modes

**Question**: what actually moves generation throughput on an AGX Orin 64GB,
and where is the physical ceiling?

Runs live in `runs/` (named configurations) and `raw/` (the untouched
`bench-run-*` directories from the session). Both were produced with
`bench/capture-jetson-env.sh --wrap`.

## Results

| Configuration | tok/s |
| --- | --- |
| 30 W stock | 8.1 |
| MAXN stock clocks | 26.9 |
| MAXN + jetson_clocks | 27.1 |
| JetPack 6.0 → 6.2.1, same build | +1% |
| IQ4_XS (same model, 10% fewer bytes) | 34.5 |
| Speculative decoding (1B draft, `--spec-draft-n-max 8`) | 43.0 |

Model: Llama 3.1 8B Q4_K_M unless stated, llama.cpp with CUDA.

Findings:

- The **bandwidth ceiling** predicts single-stream generation: memory
  bandwidth divided by model file size gives 204.8 GB/s ÷ 4.92 GB = 41.6 tok/s
  for this model. Any published number above that ceiling for this
  model/quantization is wrong or mislabelled.
- The **power mode** is the dominant knob (x3.3 on its own); `jetson_clocks`
  adds only +0.7% but divides run-to-run deviation by 16.
- **Speculative decoding is the only lever that beats the naive ceiling** at
  identical model and quantization, because the target model validates several
  tokens per weight read.

## Reproducing

```bash
sudo ../../bench/maximize-perf.sh
../../bench/capture-jetson-env.sh --wrap -- llama-bench -m <model.gguf> ...
```
