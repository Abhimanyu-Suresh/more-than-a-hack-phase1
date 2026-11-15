"""Possession labelling with a 3-state HMM (H / A / D) decoded by Viterbi.

Why: spellF1 rewards recovering whole possessions, so the output must be a few long runs, not a
flickering per-frame vote. Instead of smoothing hard labels with a majority vote (the baseline), we
accumulate *soft* evidence per frame and let Viterbi pick the single best state path, with a
switching penalty that encodes "possessions last ~10-15 s".

  evidence per frame (ball visible):  who is closer to the ball -> log-likelihood ratio H vs A
  evidence per frame (ball missing):  none -> the path simply carries on (the same as the scorer's
                                      forward-fill of X, but chosen by the decoder)
  dead ball evidence:                 ball outside the pitch, or ball stationary for a sustained time

numpy only. Run:   python solution/hmm_possession.py --out evaluation/results/hmm_v1.csv

All constants are set from first principles / the documented mean possession length, NOT tuned on the
public 5-minute window (EVALUATION.md and RULES.md forbid choosing between versions with it).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FPS = 25
HOME, AWAY, DEAD = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# ---- parameters (first-principles; see module docstring) ----
MEAN_POSSESSION_S = 14.0        # documented mean possession length in the public GT description
MEAN_DEAD_S = 8.0               # typical stoppage; only affects how sticky D is
CONTROL_RADIUS_M = 4.0          # a player must be this close to the ball to count as evidence
LLR_SCALE_M = 2.0               # distance difference (m) that gives one unit of log-odds
LLR_CLIP = 3.0
FLIGHT_SPEED_MS = 14.0          # ball faster than this is a pass/shot in flight: weak evidence
FLIGHT_WEIGHT = 0.25
STATIONARY_SPEED_MS = 0.6       # ball slower than this ...
STATIONARY_MIN_S = 3.0          # ... for this long is treated as dead
OUT_MARGIN = 0.005              # normalised units beyond the touchlines counts as out of pitch
DEAD_BONUS = 2.0                # log-evidence per frame for a "dead" frame (D vs H/A)
ALIVE_PENALTY = 0.15            # small per-frame log-cost of D when the ball is clearly live


# --------------------------------------------------------------------------------------------
# Reading the raw ATD
# --------------------------------------------------------------------------------------------
def read_atd(raw_path, n_frames, n_slots=None):
    """Stream the raw ATD into arrays. Positions are normalised 0-1; NaN where missing.

    Returns ball (n,2) and players (n, slots, 2). Line format: "<frame_id>:<p>;<p>;...:<ball>",
    each group "x,y,speed". frame_id counts from 1, submissions from 0.
    """
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = None
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            groups = parts[1].split(";")
            if players is None:
                n_slots = n_slots or len(groups)
                players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
            for slot, group in enumerate(groups[:players.shape[1]]):
                if group[:1] == "N":
                    continue
                x, y, _ = group.split(",")
                players[frame, slot, 0] = float(x)
                players[frame, slot, 1] = float(y)
            if parts[2][:1] != "N":
                x, y, _ = parts[2].split(",")
                ball[frame, 0] = float(x)
                ball[frame, 1] = float(y)
    return ball, players


# --------------------------------------------------------------------------------------------
# Per-frame evidence
# --------------------------------------------------------------------------------------------
def ball_speed(ball_m, half_blocks, window=5):
    """Ball speed in m/s from positions `window` frames apart, NaN where either end is missing."""
    speed = np.full(len(ball_m), np.nan, dtype=np.float32)
    for _, lo, hi in half_blocks:
        seg = ball_m[lo:hi + 1]
        if len(seg) <= window:
            continue
        step = np.linalg.norm(seg[window:] - seg[:-window], axis=1) / (window / FPS)
        mid = np.arange(lo + window // 2, lo + window // 2 + len(step))
        speed[mid] = step
    return speed


def stationary_flag(speed, min_frames):
    """True where the ball has been slower than STATIONARY_SPEED_MS for >= min_frames in a row.

    Missing speed (ball not seen) breaks a run: we never infer "dead" from an absent ball.
    """
    slow = np.nan_to_num(speed, nan=1e9) < STATIONARY_SPEED_MS
    out = np.zeros(len(speed), dtype=bool)
    run_start = None
    for i, s in enumerate(np.append(slow, False)):
        if s and run_start is None:
            run_start = i
        elif not s and run_start is not None:
            if i - run_start >= min_frames:
                out[run_start:i] = True
            run_start = None
    return out


def frame_log_emissions(ball, players, team_of_slot, pitch, half_blocks):
    """(n, 3) log-evidence for H, A, D per frame. Missing ball -> all zeros (uninformative)."""
    n = len(ball)
    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)
    ball_m = ball * scale
    players_m = players * scale

    speed = ball_speed(ball_m, half_blocks)
    seen = ~np.isnan(ball_m[:, 0])

    d = np.linalg.norm(players_m - ball_m[:, None, :], axis=2)        # (n, slots), NaN if missing
    d_home = np.where(team_of_slot[None, :] == HOME, d, np.inf)
    d_away = np.where(team_of_slot[None, :] == AWAY, d, np.inf)
    d_home = np.nan_to_num(d_home, nan=np.inf).min(axis=1)
    d_away = np.nan_to_num(d_away, nan=np.inf).min(axis=1)

    nearest = np.minimum(d_home, d_away)
    informative = seen & (nearest < CONTROL_RADIUS_M)
    diff = np.where(np.isfinite(d_home) & np.isfinite(d_away), d_away - d_home,
                    np.where(np.isfinite(d_home), CONTROL_RADIUS_M, -CONTROL_RADIUS_M))
    llr = np.clip(diff / LLR_SCALE_M, -LLR_CLIP, LLR_CLIP)
    weight = np.where(np.nan_to_num(speed, nan=0.0) > FLIGHT_SPEED_MS, FLIGHT_WEIGHT, 1.0)
    llr = np.where(informative, llr * weight, 0.0)

    out_of_pitch = seen & ((ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > 1 + OUT_MARGIN)
                           | (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > 1 + OUT_MARGIN))
    still = stationary_flag(speed, int(STATIONARY_MIN_S * FPS))
    dead = out_of_pitch | still
    live = seen & ~dead & (np.nan_to_num(speed, nan=0.0) > 2.0)

    em = np.zeros((n, 3), dtype=np.float64)
    em[:, HOME] = 0.5 * llr
    em[:, AWAY] = -0.5 * llr
    em[dead, DEAD] = DEAD_BONUS
    em[dead, HOME] -= DEAD_BONUS / 2
    em[dead, AWAY] -= DEAD_BONUS / 2
    em[live, DEAD] -= ALIVE_PENALTY
    return em


# --------------------------------------------------------------------------------------------
# Viterbi
# --------------------------------------------------------------------------------------------
def transition_matrix():
    p_leave = 1.0 / (MEAN_POSSESSION_S * FPS)
    p_dead_in = 0.25 * p_leave                       # H/A -> D (stoppages are rarer than turnovers)
    p_switch = p_leave - p_dead_in
    p_dead_out = 1.0 / (MEAN_DEAD_S * FPS)
    trans = np.array([
        [1 - p_leave, p_switch, p_dead_in],
        [p_switch, 1 - p_leave, p_dead_in],
        [p_dead_out / 2, p_dead_out / 2, 1 - p_dead_out],
    ])
    return np.log(trans)


def viterbi(em, log_trans):
    """Best state path for (T, 3) log-emissions. Starts uniform."""
    T = len(em)
    score = np.empty((T, 3))
    back = np.empty((T, 3), dtype=np.int8)
    score[0] = em[0]
    for t in range(1, T):
        cand = score[t - 1][:, None] + log_trans          # (from, to)
        back[t] = np.argmax(cand, axis=0)
        score[t] = cand[back[t], np.arange(3)] + em[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(em, half_blocks, n_frames):
    """Run Viterbi per half; frames outside the halves are X (never scored)."""
    labels = np.full(n_frames, "X", dtype="<U1")
    log_trans = transition_matrix()
    for _, lo, hi in half_blocks:
        labels[lo:hi + 1] = LABELS[viterbi(em[lo:hi + 1], log_trans)]
    return labels


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def team_slots(meta_path):
    from baseline import team_of_each_track      # same convention as the starter
    return team_of_each_track(meta_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "hmm_v1.csv"))
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n_frames = source_frames("tactical", GAME)
    blocks = half_video_blocks(GAME)
    print("reading tracking ...")
    ball, players = read_atd(raw_path, n_frames)
    team = team_slots(meta_path)
    print("ball seen on {:.1f}% of frames".format(100 * np.mean(~np.isnan(ball[:, 0]))))
    em = frame_log_emissions(ball, players, team, GAMES[GAME]["pitch"], blocks)
    labels = decode(em, blocks, n_frames)
    write_submission(args.out, labels)
    for lab in "HAD":
        print(lab, int(np.sum(labels == lab)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()