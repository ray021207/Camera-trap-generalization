"""Phase 7: fill the README results table straight from results/*.json (no hand-typed numbers).

    python scripts/12_make_tables.py
Replaces the text between the RESULTS_TABLE markers in README.md and also writes results/tables.md.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, load_json


def cell(s, k):
    lo, hi = s[k + "_ci95"]
    return f"{s[k]:.3f} [{lo:.3f}, {hi:.3f}]"


def gap_cell(g, k):
    lo, hi = g[k + "_ci95"]
    return f"{g[k]:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    f, c = load_json("phase5_eval_full"), load_json("phase5_eval_crop")
    p6 = load_json("phase6_mitigation")
    rows = [("(a) random image split (seen cameras)", "a_random_image"),
            ("(b) random sequence split (seen cameras)", "b_random_sequence"),
            ("(c) held-out cameras (CCT20 trans_test)", "c_unseen_locations")]
    t = ["| Split | Accuracy, baseline | Macro-F1, baseline | Accuracy, MegaDetector crop | Macro-F1, MegaDetector crop |",
         "|---|---|---|---|---|"]
    for name, k in rows:
        t.append(f"| {name} | {cell(f['setups'][k], 'accuracy')} | {cell(f['setups'][k], 'macro_f1')} | "
                 f"{cell(c['setups'][k], 'accuracy')} | {cell(c['setups'][k], 'macro_f1')} |")
    for name, k in [("Gap (b) minus (c): the camera-shift penalty", "b_seen_minus_c_unseen"),
                    ("Gap (a) minus (c): leakage + camera shift", "a_random_image_minus_c_unseen")]:
        t.append(f"| **{name}** | {gap_cell(f['gaps'][k], 'accuracy_gap')} | {gap_cell(f['gaps'][k], 'macro_f1_gap')} | "
                 f"{gap_cell(c['gaps'][k], 'accuracy_gap')} | {gap_cell(c['gaps'][k], 'macro_f1_gap')} |")
    d = p6["deltas"]
    t += ["", "Paired change from cropping (crop minus baseline), 95% CI from the same bootstrap resamples:", "",
          "| Quantity | Accuracy | Macro-F1 |", "|---|---|---|"]
    for name, key in [("Unseen-camera score", "{}_unseen_change_crop_minus_full"), ("Seen-camera score", "{}_seen_change_crop_minus_full"),
                      ("Seen-minus-unseen gap", "{}_gap_change_crop_minus_full")]:
        a, m = d[key.format("accuracy")], d[key.format("macro_f1")]
        t.append(f"| {name} | {a['value']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {m['value']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}] |")
    ctl = load_json("phase6_control_mask")
    t += ["", "Control: is it the background, or the zoom? (paired 95% CI; same frozen backbone and probe)", "",
          "| Input to the model | Unseen-camera accuracy | Unseen-camera macro-F1 | Seen-minus-unseen accuracy gap |", "|---|---|---|---|"]
    for name, v in [("whole frame (baseline)", "full"), ("whole frame, background greyed out (same resolution)", "mask"), ("MegaDetector crop", "crop")]:
        r = ctl["variants"][v]
        g = r["accuracy_gap_seen_minus_unseen"]
        t.append(f"| {name} | {cell(r['unseen'], 'accuracy')} | {cell(r['unseen'], 'macro_f1')} | {g['value']:.3f} [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}] |")
    t += ["", f"Test sizes: (a) {f['setups']['a_random_image']['n_images']} images, (b) {f['setups']['b_random_sequence']['n_images']} images, "
              f"(c) {f['setups']['c_unseen_locations']['n_images']} images in {f['setups']['c_unseen_locations']['n_sequences']} bursts. "
              f"CIs: {f['ci_method']}, {f['n_boot']} resamples."]
    block = "\n".join(t)
    (ROOT / "results" / "tables.md").write_text(block + "\n", encoding="utf-8")
    rd = (ROOT / "README.md").read_text(encoding="utf-8")
    new = re.sub(r"(<!-- RESULTS_TABLE:START -->)(.*?)(<!-- RESULTS_TABLE:END -->)", lambda m: m.group(1) + "\n" + block + "\n" + m.group(3), rd, flags=re.S)
    (ROOT / "README.md").write_text(new, encoding="utf-8")
    print(block)


if __name__ == "__main__":
    main()
