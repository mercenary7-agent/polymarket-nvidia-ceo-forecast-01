#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry.

Reads each case file <id>.in in the input dir (canonical JSON, one market per
entry with its 1-minute Yes-price grid up to the checkpoint) and writes
<id>.out: for every market, 240 rows of the five quantiles (0.05..0.95) of
the price at minutes 1..240 after the checkpoint.

Model: per market, blend the realized volatility of the last 120 and last
360 one-minute returns, spread it over the horizon as sigma * sqrt(h)^1.1,
center the median on the last price, and emit normal quantiles clamped to
[0,1] and rounded to 6 decimals. Fully deterministic, stdlib only.
"""
import json
import math
import os

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

# blended-window volatility weights (short window, long window)
W_SHORT, C_SHORT = 120, 0.35
W_LONG, C_LONG = 360, 0.20
POW_H = 1.1  # horizon spread exponent, tuned on the practice cases


def price6(x):
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0  # normalize -0.0


def ret_sd(prices, w):
    p = prices[-w:] if 0 < w < len(prices) else prices
    n = len(p)
    if n < 3:
        return 0.0
    rets = [p[i + 1] - p[i] for i in range(n - 1)]
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def forecast(market, horizon):
    prices = market["prices"]
    x = prices[-1]
    sigma = (C_SHORT * ret_sd(prices, W_SHORT) + C_LONG * ret_sd(prices, W_LONG)) / 2.0
    rows = []
    for h in range(1, horizon + 1):
        s = sigma * (math.sqrt(h) ** POW_H)
        rows.append([price6(x + z * s) for z in Z])
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        cid = name[: -len(".in")]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = case["horizon_minutes"]
        forecasts = [{"id": m["id"], "q": forecast(m, horizon)} for m in case["markets"]]
        with open(os.path.join(out_dir, cid + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
