# DFR MVTec AD reproduction results

Reported categories: 8/15; categories at the current 1-epoch target: 8/15.

| Category | Epochs | PCA dim | Train (h) | Eval (s) | Det AP | Det AUC | Seg AP | Seg AUC | PRO-AUC | Best IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 1 | 197 | 0.000 | 0.0 | 0.90654 | 0.73810 | 0.16946 | 0.66200 | 0.50997 | 0.17431 |
| cable | 1 | 669 | 0.000 | 0.0 | 0.79231 | 0.64074 | 0.07621 | 0.75868 | 0.42493 | 0.11372 |
| capsule | 1 | 262 | 0.000 | 0.0 | 0.80151 | 0.42361 | 0.06978 | 0.90979 | 0.79052 | 0.07016 |
| hazelnut | 1 | 514 | 0.000 | 0.0 | 0.99923 | 0.99857 | 0.59686 | 0.97772 | 0.93208 | 0.44586 |
| metal_nut | 1 | 532 | 0.000 | 0.0 | 0.85217 | 0.60899 | 0.21770 | 0.69898 | 0.40765 | 0.22219 |
| carpet | 1 | 383 | 0.000 | 0.0 | 0.96954 | 0.89968 | 0.29617 | 0.87124 | 0.74199 | 0.24655 |
| grid | 1 | 209 | 0.000 | 0.0 | 0.87035 | 0.69674 | 0.01165 | 0.53132 | 0.26162 | 0.04187 |
| leather | 1 | 432 | 0.000 | 0.0 | 0.97301 | 0.92018 | 0.41401 | 0.95938 | 0.86611 | 0.28927 |
| **Mean metrics** |  |  |  |  | **0.89558** | **0.74083** | **0.23148** | **0.79614** | **0.61686** | **0.20049** |

A zero timing value marks a smoke result produced before cumulative timing metadata was introduced.

The upstream project does not publish complete package versions or all random-state details; differences from the paper are reported without hidden tuning.
