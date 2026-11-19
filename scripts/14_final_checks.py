"""Phase 7: verify the 'done means' list and record it in results/phase7_final_checks.json.

    python scripts/14_final_checks.py

Checks: every expected results file exists; folder size vs the 6 GB budget; the private study guide (if present locally)
has all sub-headings for every phase; README table is filled; .gitignore entries; and that neither the tracked
files nor the git history contain any AI-attribution strings.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, save_json

EXPECTED = ["phase0", "phase1_sampling", "phase1_download", "phase1_audit", "phase1_full_cct_context", "phase2_splits", "phase3",
            "phase4_extract_full", "phase4_probe_full", "phase4_umap_full", "phase5_eval_full", "phase5_failures",
            "phase4_extract_crop", "phase4_probe_crop", "phase4_umap_crop", "phase5_eval_crop", "phase6_mitigation",
            "phase4_extract_mask", "phase4_probe_mask", "phase4_umap_mask", "phase5_eval_mask", "phase6_control_mask", "licenses"]
SUBHEADS = ["What we did", "Why we did it", "How it works", "Why not the alternatives", "The numbers we got", "Why we got these numbers", "Try it yourself"]
BAD = re.compile(r"Claude|Anthropic|Co-Authored|Generated with|ChatGPT|Copilot|OpenAI", re.I)


def folder_gb(skip=(".git",)):
    tot = 0
    for dp, dn, fs in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in skip]
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot / 1e9


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout


def main():
    out = {}
    out["expected_results_files_missing"] = [n for n in EXPECTED if not (ROOT / "results" / f"{n}.json").exists()]
    out["n_json_files_in_results"] = len(list((ROOT / "results").glob("*.json")))
    out["folder_gb_excluding_dot_git"] = round(folder_gb(), 3)
    out["data_dir_gb"] = round(sum(f.stat().st_size for f in (ROOT / "data").rglob("*") if f.is_file()) / 1e9, 3)
    out["venv_gb"] = round(sum(f.stat().st_size for f in (ROOT / ".venv").rglob("*") if f.is_file()) / 1e9, 3)
    out["budget_gb"] = 6.0
    out["under_budget"] = out["folder_gb_excluding_dot_git"] < 6.0
    notes_path = ROOT / "NOTES.md"      # private study guide: kept out of git; checked only when present
    if notes_path.exists():
        notes = notes_path.read_text(encoding="utf-8")
        phases = {sec.split("
")[0]: sec for sec in re.split(r"
## ", notes) if sec.startswith("Phase ")}
        missing = {p: [h for h in SUBHEADS if f"### {h}" not in body] for p, body in phases.items()}
        out["private_notes_all_phases_have_all_subheadings"] = not any(missing.values()) and len(phases) == 8
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"RESULTS_TABLE:START -->(.*?)<!-- RESULTS_TABLE:END", readme, re.S)
    out["readme_results_table_filled"] = bool(m and "|" in m.group(1))
    gi = (ROOT / ".gitignore").read_text()
    out["gitignore_has_required_entries"] = all(x in gi for x in [".claude/", "CLAUDE.md", "data/"])
    files = [f for f in git("ls-files").splitlines() if not f.startswith("licenses/") and f not in ("scripts/14_final_checks.py", ".gitignore")]  # .gitignore must name CLAUDE.md (required ignore entry), not attribution
    hits = []
    for f in files:
        try:
            txt = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if BAD.search(txt):
            hits.append(f)
    out["tracked_files_with_ai_attribution_strings"] = hits
    log = git("log", "--format=%an <%ae>%n%B")
    out["git_log_ai_attribution_hits"] = len(BAD.findall(log))
    out["n_commits"] = len(git("log", "--format=%H").split())
    out["commit_authors"] = sorted(set(git("log", "--format=%an <%ae>").strip().splitlines()))
    save_json(out, "phase7_final_checks")
    print({k: v for k, v in out.items()})


if __name__ == "__main__":
    main()
