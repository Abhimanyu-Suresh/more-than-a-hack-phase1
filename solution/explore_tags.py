import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load as load_tags

lab = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
L = lambda i: lab[i] if 0 <= i < len(lab) else "-"

events = [e for e in load_tags() if e["end_frame"] >= 11185 - 250 and e["start_frame"] <= 18684 + 250]
for e in sorted(events, key=lambda e: e["start_frame"]):
    a, b = e["start_frame"], e["end_frame"]
    print("{:<30} {:<12} {:<13} {:>6}-{:<6} {:>5.1f}s | GT @start-250,start,end,end+250: {} {} {} {}".format(
        e["code"], str(e["tags"].get("Type")), str(e["tags"].get("Team")), a, b, (b - a) / 25,
        L(a - 250), L(a), L(b), L(b + 250)))