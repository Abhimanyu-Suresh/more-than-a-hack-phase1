"""v3: same model as v2, but the two key strengths are parameters and can be swept.

  --sweep                      decode over a grid and print possession count / mean duration / D share
                               for the FULL match (no ground truth involved)
  --scale S --dead-mult M      write a submission with those settings

S      = evidence scale (higher -> team switches need less evidence -> more, shorter possessions)
M      = multiplier on all dead-ball evidence (higher -> more D)

Pick settings by plausibility of the full-match shape (published public-window stats: possessions
last ~14 s, dead ball ~20% of graded frames), NOT by the public-window score (17 possessions cannot
rank anything -- see EVALUATION.md).

    python solution/hmm_possession_v3.py --sweep
    python solution/hmm_possession_v3.py --scale 0.2 --dead-mult 0.6 --out evaluation/results/hmm_v3.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import hmm_possession as h  # noqa: E402
import hmm_possession_v2 as v2  # noqa: E402

HOME, AWAY, DEAD = 0, 1, 2


def components(ball, players, team, pitch, blocks, events):
    n = len(ball)
    scale = np.array(pitch, dtype=np.float32)
    ball_m = ball * scale
    seen = ~np.isnan(ball_m[:, 0])
    speed = h.ball_speed(ball_m, blocks)

    em_v1 = h.frame_log_emissions(ball, players, team, pitch, blocks)
    team_ev = em_v1[:, :2].copy()
    dead_v1 = em_v1[:, DEAD] >= h.DEAD_BONUS - 1e-9
    team_ev[dead_v1] += h.DEAD_BONUS / 2          # undo v1's dead-frame H/A penalty

    out_of_pitch = seen & ((ball[:, 0] < -h.OUT_MARGIN) | (ball[:, 0] > 1 + h.OUT_MARGIN)
                           | (ball[:, 1] < -h.OUT_MARGIN) | (ball[:, 1] > 1 + h.OUT_MARGIN))
    gaps = v2.boundary_gaps(ball_m, pitch, blocks)
    live = seen & ~out_of_pitch & (np.nan_to_num(speed, nan=0.0) > v2.LIVE_SPEED_MS)
    dead_raw = np.zeros(n)
    dead_raw[out_of_pitch] += v2.OUT_OF_PITCH_D
    dead_raw[gaps] += v2.BOUNDARY_GAP_D
    dead_raw[live] -= v2.LIVE_D_PENALTY

    team_prior, dead_prior = v2.tag_priors(events, n)
    return team_ev, dead_raw, team_prior, dead_prior


def build_em(comp, scale, dead_mult):
    team_ev, dead_raw, team_prior, dead_prior = comp
    em = np.zeros((len(team_ev), 3))
    em[:, :2] = scale * team_ev
    em[:, DEAD] = scale * dead_raw * dead_mult
    em += team_prior
    em[:, DEAD] += dead_prior * dead_mult
    return em


def shape(labels, blocks):
    graded = sum(hi - lo + 1 for _, lo, hi in blocks)
    d_share = 100.0 * sum(int(np.sum(labels[lo:hi + 1] == "D")) for _, lo, hi in blocks) / graded
    runs, mean_s = v2.summarise(labels, blocks)
    return d_share, runs, mean_s


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--dead-mult", type=float, default=0.6)
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "hmm_v3.csv"))
    args = parser.parse_args()

    from load_smart_tagging import load as load_tags
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n_frames = source_frames("tactical", GAME)
    blocks = half_video_blocks(GAME)
    print("reading tracking ...")
    ball, players = h.read_atd(raw_path, n_frames)
    team = h.team_slots(meta_path)
    comp = components(ball, players, team, GAMES[GAME]["pitch"], blocks, load_tags())

    if args.sweep:
        print("\n scale  dead_mult |  D%   possessions  mean_s      (target: D ~20%, ~350-400, ~14 s)")
        for scale in (0.08, 0.12, 0.18, 0.25, 0.35, 0.5):
            for dm in (1.0, 0.6, 0.35):
                labels = h.decode(build_em(comp, scale, dm), blocks, n_frames)
                d, runs, mean_s = shape(labels, blocks)
                print(" {:>5}  {:>8}  | {:>4.1f}  {:>8}   {:>6.1f}".format(scale, dm, d, runs, mean_s))
        return

    labels = h.decode(build_em(comp, args.scale, args.dead_mult), blocks, n_frames)
    h.write_submission(args.out, labels)
    d, runs, mean_s = shape(labels, blocks)
    print("scale {} dead_mult {}: D {:.1f}%  possessions {}  mean {:.1f} s".format(
        args.scale, args.dead_mult, d, runs, mean_s))
    print("wrote", args.out)


if __name__ == "__main__":
    main()