# §3.3 cost-ordering panel — sftv22, K=8, groups=799

reward replay: retrieval_cost=0.015, token_cost=0.001, format_penalty=0.2, turn_limit_penalty=0.3
n_search 口径核对: 0/6392 rows 与 hops 记录不一致

| slice | all-failure groups | mean within-group rho(A, n_search) | share rho<0 | mean rho(A, coverage) |
|---|---|---|---|---|
| all hops | 400 | -0.913 (308 def, 92 const) | 97% | +0.024 (236 def) |
| 2-hop | 68 | -0.936 (56 def, 12 const) | 98% | -0.103 (24 def) |
| 3-hop | 149 | -0.941 (126 def, 23 const) | 99% | +0.050 (88 def) |
| 4-hop | 183 | -0.874 (126 def, 57 const) | 95% | +0.030 (124 def) |

all-failure rollouts: 3200; gave-up 1738 (54.3%), with-invalid-turn 23 (0.7%)
