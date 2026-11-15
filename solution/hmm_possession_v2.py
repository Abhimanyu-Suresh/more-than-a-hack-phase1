"""Possession labelling v2: HMM (H / A / D) + Viterbi, with Smart Tagging priors and dead-ball cues.

Changes from v1 (see hmm_possession.py):
  1. Per-frame evidence is scaled down: at 25 fps consecutive frames are nearly the same
     observation, so treating each as independent made team switches far too cheap.
  2. Dead-ball evidence: ball out of the pitch; long ball gaps that start or end near a boundary
     (the ball left play and was not tracked); Smart Tagging set-piece windows. The v1
     "stationary ball" rule is dropped (it was wrong: dead balls are slow, not still).
  3. Smart Tagging team priors: attacking-phase tags favour that team, defending tags the other
     team, a set piece's team gets a prior after the restart. Tags are inferred, so priors are weak.

Run:   python solution/hmm_possession_v2.py --out evaluation/results/hmm_v2.csv

Constants are chosen from football reasoning, not tuned on the public 5-minute window.
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

FPS = 25
HOME, AWAY, DEAD = 0, 1, 2

# ---- evidence (raw units, multiplied by EVIDENCE_SCALE) ----
EVIDENCE_SCALE = 0.08            # ~1 independent observation per 0.5 s
OUT_OF_PITCH_D = 3.0             # raw D evidence per frame, ball outside the pitch
BOUNDARY_GAP_D = 1.5             # raw D evidence per frame inside a long ball gap near a boundary
GAP_MIN_S = 3.0                  # ball unseen at least this long ...
GAP_BOUNDARY_M = 4.0             # ... and last/next seen within this distance of a boundary
LIVE_SPEED_MS = 5.0              # clearly moving, in-pitch ball: small evidence against D
LIVE_D_PENALTY = 0.3

# ---- Smart Tagging priors (post-scale, per frame) ----
SETPIECE_D = 0.10                # D evidence inside a set-piece window
SETPIECE_PRE_S, SETPIECE_POST_S = 4.0, 8.0     # window around the tag start (restart is fuzzy)
SETPIECE_TEAM_AFTER_S = 6.0      # team prior starts this long after the tag start
PHASE_TEAM = 0.02                # attacking / defending phase tag team prior
SETPIECE_TEAM = 0.03             # restarting team prior after the restart

ATTACK_CODES = {"progression", "build up", "attacking transition", "creating chances",
                "ball in final third"}
DEFEND_CODES = {"defending in middle third", "defending in defensive third",
                "defending in attacking third", "defensive transition"}


def side_of(team_name):
    """'Barcelona' -> HOME, 'Atletic Club' -> AWAY, anything else (N/A) -> None."""
    name = (team_name or "").strip().lower()
    if "barcelona" in name:
        return HOME
    if "atletic" in name or "athletic" in name:
        return AWAY
    return None


def tag_priors(events, n_frames):
    """Return (team_prior (n,3), dead_prior (n,)): post-scale log-evidence from Smart Tagging."""
    team = np.zeros((n_frames, 3))
    dead = np.zeros(n_frames)
    for e in events:
        code = (e["code"] or "").strip().lower()
        a = max(0, e["start_frame"])
        b = min(n_frames, e["end_frame"])
        if b <= a:
            continue
        side = side_of(e["tags"].get("Team"))
        if code == "set pieces":
            kind = (e["tags"].get("Type") or "").strip().lower()
            if kind != "kick off (start)":
                lo = max(0, a - int(SETPIECE_PRE_S * FPS))
                hi = min(n_frames, a + int(SETPIECE_POST_S * FPS))
                dead[lo:hi] += SETPIECE_D
            if side is not None:
                lo = min(n_frames, a + int(SETPIECE_TEAM_AFTER_S * FPS))
                if lo < b:
                    team[lo:b, side] += SETPIECE_TEAM
        elif side is not None and code in ATTACK_CODES:
            team[a:b, side] += PHASE_TEAM
        elif side is not None and code in DEFEND_CODES:
            team[a:b, 1 - side] += PHASE_TEAM
    return team, dead


def boundary_distance(ball_m, pitch):
    length, width = pitch
    x, y = ball_m[..., 0], ball_m[..., 1]
    return np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))


def boundary_gaps(ball_m, pitch, half_blocks):
    """Boolean per frame: inside a ball-missing run >= GAP_MIN_S whose edge is near a boundary."""
    n = len(ball_m)
    flag = np.zeros(n, dtype=bool)
    seen = ~np.isnan(ball_m[:, 0])
    dist = boundary_distance(ball_m, pitch)
    min_len = int(GAP_MIN_S * FPS)
    for _, lo, hi in half_blocks:
        i = lo
        while i <= hi:
            if seen[i]:
                i += 1
                continue
            j = i
            while j <= hi and not seen[j]:
                j += 1
            if j - i >= min_len:
                edges = []
                if i - 1 >= lo and seen[i - 1]:
                    edges.append(dist[i - 1])
                if j <= hi and seen[j]:
                    edges.append(dist[j])
                if edges and min(edges) < GAP_BOUNDARY_M:
                    flag[i:j] = True
            i = j
    return flag


def frame_log_emissions_v2(ball, players, team_of_slot, pitch, half_blocks, events):
    n = len(ball)
    scale = np.array(pitch, dtype=np.float32)
    ball_m = ball * scale
    seen = ~np.isnan(ball_m[:, 0])
    speed = h.ball_speed(ball_m, half_blocks)

    # team evidence from proximity (same as v1, minus the dead rules)
    em_v1 = h.frame_log_emissions(ball, players, team_of_slot, pitch, half_blocks)
    team_ev = np.zeros((n, 3))
    team_ev[:, HOME] = em_v1[:, HOME]
    team_ev[:, AWAY] = em_v1[:, AWAY]
    # v1 subtracted from H/A on its dead frames; undo by recomputing without the dead part
    dead_v1 = em_v1[:, DEAD] >= h.DEAD_BONUS - 1e-9
    team_ev[dead_v1, HOME] += h.DEAD_BONUS / 2
    team_ev[dead_v1, AWAY] += h.DEAD_BONUS / 2

    out_of_pitch = seen & ((ball[:, 0] < -h.OUT_MARGIN) | (ball[:, 0] > 1 + h.OUT_MARGIN)
                           | (ball[:, 1] < -h.OUT_MARGIN) | (ball[:, 1] > 1 + h.OUT_MARGIN))
    gaps = boundary_gaps(ball_m, pitch, half_blocks)
    live = seen & ~out_of_pitch & (np.nan_to_num(speed, nan=0.0) > LIVE_SPEED_MS)

    dead_raw = np.zeros(n)
    dead_raw[out_of_pitch] += OUT_OF_PITCH_D
    dead_raw[gaps] += BOUNDARY_GAP_D
    dead_raw[live] -= LIVE_D_PENALTY

    em = EVIDENCE_SCALE * team_ev
    em[:, DEAD] += EVIDENCE_SCALE * dead_raw

    team_prior, dead_prior = tag_priors(events, n)
    em += team_prior
    em[:, DEAD] += dead_prior
    return em


def summarise(labels, half_blocks):
    """Possession count/duration the way the scorer counts them (runs of H or A, D splits)."""
    runs, lengths = 0, []
    for _, lo, hi in half_blocks:
        seg = labels[lo:hi + 1]
        start = 0
        for t in range(1, len(seg) + 1):
            if t == len(seg) or seg[t] != seg[start]:
                if seg[start] in ("H", "A"):
                    runs += 1
                    lengths.append((t - start) / FPS)
                start = t
    return runs, (float(np.mean(lengths)) if lengths else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "hmm_v2.csv"))
    args = parser.parse_args()

    from load_smart_tagging import load as load_tags
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n_frames = source_frames("tactical", GAME)
    blocks = half_video_blocks(GAME)
    print("reading tracking ...")
    ball, players = h.read_atd(raw_path, n_frames)
    team = h.team_slots(meta_path)
    events = load_tags()
    print("ball seen on {:.1f}% of frames; {} tag instances".format(
        100 * np.mean(~np.isnan(ball[:, 0])), len(events)))

    em = frame_log_emissions_v2(ball, players, team, GAMES[GAME]["pitch"], blocks, events)
    labels = h.decode(em, blocks, n_frames)
    h.write_submission(args.out, labels)

    graded = sum(hi - lo + 1 for _, lo, hi in blocks)
    for lab in "HAD":
        k = int(np.sum(labels == lab))
        print("{} {:>7}  ({:.1f}% of graded)".format(lab, k, 100 * k / graded))
    runs, mean_s = summarise(labels, blocks)
    print("possessions (runs of H/A): {}  mean {:.1f} s   [expect roughly 350-400, ~14 s]".format(
        runs, mean_s))
    print("wrote", args.out)


if __name__ == "__main__":
    main()