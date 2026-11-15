import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "starter")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from games import GAME, GAMES, half_video_blocks, require, source_frames
import hmm_possession as h

meta, raw = require(["atd_meta", "atd_raw"], GAME, None)
n = source_frames("tactical", GAME)
ball, players = h.read_atd(raw, n)
blocks = half_video_blocks(GAME)
scale = np.array(GAMES[GAME]["pitch"], dtype=np.float32)
speed = h.ball_speed(ball * scale, blocks)

gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
lab = gt["label"].values
sel = np.where(lab != "X")[0]
print("graded public frames:", len(sel))

seen = ~np.isnan(ball[:, 0])
out = seen & ((ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1))
n_players = np.sum(~np.isnan(players[:, :, 0]), axis=1)

rows = []
for name, mask in (("D (dead)", lab == "D"), ("H/A (alive)", (lab == "H") | (lab == "A"))):
    m = mask & (np.arange(n) >= sel[0]) & (np.arange(n) <= sel[-1])
    rows.append({
        "group": name, "frames": int(m.sum()),
        "ball_missing %": round(100 * np.mean(~seen[m]), 1),
        "ball_out_of_pitch %": round(100 * np.mean(out[m]), 1),
        "speed<0.6 %": round(100 * np.nanmean(speed[m] < 0.6), 1),
        "median speed": round(float(np.nanmedian(speed[m])), 2),
        "median players": float(np.median(n_players[m])),
    })
print(pd.DataFrame(rows).to_string(index=False))

# dead-ball segments: start, length (s), ball missing share
d = (lab == "D").astype(int)
edges = np.flatnonzero(np.diff(np.concatenate(([0], d, [0]))))
print("\nD segments (start_frame, length_s, ball_missing %):")
for a, b in zip(edges[::2], edges[1::2]):
    print(a, round((b - a) / 25, 1), round(100 * np.mean(~seen[a:b]), 0))