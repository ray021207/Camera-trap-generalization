"""Local demo: the same app as the portfolio website (space/app.py), run on this machine.

    python scripts/demo_app.py            # opens http://127.0.0.1:7860 in your browser
    python scripts/demo_app.py --share    # also prints a temporary public gradio.live link (anyone with it can upload)

Example buttons appear after `python scripts/build_space.py` has pre-computed them (they live in build/space/);
uploads work either way.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
built = ROOT / "build" / "space" / "app.py"
app = built if built.exists() else ROOT / "space" / "app.py"
sys.argv[0] = str(app)
runpy.run_path(str(app), run_name="__main__")
