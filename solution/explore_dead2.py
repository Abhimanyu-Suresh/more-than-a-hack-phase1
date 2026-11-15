import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "starter")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from games import GAME, GAMES, half_video_blocks, require, source_frames
import hmm_possession as h
from load_smart_tagging import load as load_tags

meta, raw = require(["atd_meta", "atd_raw"], GAME, None)
n = source_frames("tactical", GAME)
ball, players = h.read_atd(raw, n)
scale = np.array(GAMES[GAME]["pitch"], dtype=np.float32)
ball_m = ball * scale
speed = h.ball_speed(ball_m, half_blocks := half_video_blocks(GAME))

lab = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
d = (lab == "D").astype(int)
edges = np.flatnonzero(np.diff(np.concatenate(([0], d, [0]))))
tags = [e for e in load_tags() if (e["code"] or "").strip().lower() == "set pieces"]

print("ALL dead segments in the public window:")
for a, b in zip(edges[::2], edges[1::2]):
    seen = ~np.isnan(ball_m[a:b, 0])
    first = ball_m[a:b][seen][0].round(1) if seen.any() else None
    last = ball_m[a:b][seen][-1].round(1) if seen.any() else None
    ms = round(float(np.nanmean(speed[a:b])), 2) if np.any(~np.isnan(speed[a:b])) else None
    near = [(e["tags"].get("Type"), e["tags"].get("Team"), round((e["start_frame"] - a) / 25, 1))
            for e in tags if a - 250 <= e["start_frame"] <= b + 250]
    print("start", a, "len_s", round((b - a) / 25, 1), "| before", lab[a - 1], "after", lab[b] if b < len(lab) else "-",
          "| ball first/last (m)", first, last, "| mean speed", ms)
    print("     set pieces within +-10 s (type, team, start rel. to seg start in s):", near)