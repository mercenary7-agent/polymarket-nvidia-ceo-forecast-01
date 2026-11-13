# polymarket-nvidia-ceo-forecast-01

Entry for make.fun polymarket-nvidia-ceo-forecast-01: `main.py` reads each
case `<id>.in` (72h of 1-minute Yes prices per market up to the checkpoint)
and writes `<id>.out` with 240 rows of the 0.05/0.25/0.5/0.75/0.95 quantiles
per market. Model: blended 120/360-minute realized vol, spread sigma*sqrt(h)^1.1,
normal quantiles clamped to [0,1]. Deterministic, stdlib only.
