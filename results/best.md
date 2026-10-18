## Profile x concurrency (TTFA p50 ms / audible-stall % / RTF; n in parentheses when 1088)

| profile | c=1 | c=4 | c=16 | c=24 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|
| a_ramp | 49 / 0.0% / 0.18 | 61 / 0.0% / 0.24 | 91 / 0.0% / 0.41 | – | 117 / 44.7% / 0.55 | 230 / 99.0% / 0.70 |
| adaptive | 48 / 0.0% / 0.18 | – | 109 / 0.0% / 0.41 | – | 169 / 4.0% / 0.55 | 346 / 12.7% / 0.70 |
| adaptive128 | – | – | – | – | 219 / 0.0% / 0.56 | 349 / 2.7% / 0.70 |
| adaptive2 | – | – | 88 / 0.0% / 0.41 | 149 / 0.0% / 0.49 | 222 / 0.0% / 0.56 | 350 / 2.3% / 0.70 |
| adaptive2b | 49 / 0.0% / 0.18 | – | – | – | – | 349 / 10.0% / 0.70 |
| b_admission | 48 / 0.0% / 0.18 | – | 89 / 0.0% / 0.41 | – | – | – |
| c_topk | 48 / 0.0% / 0.18 (1088) | 61 / 0.0% / 0.25 | 88 / 0.0% / 0.41 (1088) | – | 117 / 63.6% / 0.56 (1088) | 2463 / 59.7% / 1.09 |
| e_ramp3 | 56 / 0.0% / 0.18 | 71 / 0.0% / 0.25 | 111 / 0.0% / 0.41 | – | 144 / 15.3% / 0.55 | 2515 / 22.3% / 1.11 |
| f_mid | 64 / 0.0% / 0.18 | 82 / 0.0% / 0.24 | 126 / 0.0% / 0.41 | – | 169 / 1.7% / 0.55 | 281 / 91.0% / 0.70 |
| stock | 80 / 0.0% / 0.18 | 102 / 0.0% / 0.24 | 162 / 0.0% / 0.41 | – | 219 / 0.0% / 0.55 | 352 / 1.3% / 0.70 |
| H200 stock (#8933) | 66 ms | 75 ms | 104 ms | – | 135 ms | 198 ms |

## Best per concurrency (audible stalls <= 1.0%)

| c | best profile | TTFA p50 | TTFA p95 | RTF | strict / audible stalls | runner-up |
|---:|---|---:|---:|---:|---:|---|
| 1 | **c_topk** | 48 ms | 50 ms | 0.178 | 0.0 / 0.0 % | adaptive (48 ms) |
| 4 | **a_ramp** | 61 ms | 81 ms | 0.244 | 0.0 / 0.0 % | c_topk (61 ms) |
| 16 | **adaptive2** | 88 ms | 144 ms | 0.408 | 10.0 / 0.0 % | c_topk (88 ms) |
| 24 | **adaptive2** | 149 ms | 232 ms | 0.495 | 2.7 / 0.0 % | – |
| 32 | **stock** | 219 ms | 328 ms | 0.547 | 0.0 / 0.0 % | adaptive128 (219 ms) |
| 64 | (none within budget) | | | | | |

## One-deploy pick (admissible at every measured level)

- none: every fully-measured profile breaks the stall budget at some level (this is the case for a load-adaptive ramp)
