# DFR MVTec AD reproduction results

Completed categories: 6/15; epochs per category: 1.

| Category | Det AP | Det AUC | Seg AP | Seg AUC | PRO-AUC | Best IoU |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 0.90654 | 0.73810 | 0.16946 | 0.66200 | 0.50997 | 0.17431 |
| cable | 0.79231 | 0.64074 | 0.07621 | 0.75868 | 0.42493 | 0.11372 |
| capsule | 0.80151 | 0.42361 | 0.06978 | 0.90979 | 0.79052 | 0.07016 |
| hazelnut | 0.99923 | 0.99857 | 0.59686 | 0.97772 | 0.93208 | 0.44586 |
| carpet | 0.96954 | 0.89968 | 0.29617 | 0.87124 | 0.74199 | 0.24655 |
| grid | 0.87035 | 0.69674 | 0.01165 | 0.53132 | 0.26162 | 0.04187 |
| **Mean** | **0.88991** | **0.73291** | **0.20336** | **0.78512** | **0.61019** | **0.18208** |

The upstream project does not publish complete package versions or all random-state details; differences from the paper are reported without hidden tuning.
