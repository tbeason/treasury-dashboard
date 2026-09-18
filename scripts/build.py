"""Build the dashboard into site/.

    uv run scripts/build.py [--open]

Set FRED_API_KEY for the primary core-CPI source (the build falls back to the
BLS public endpoint without it). TSYFWD_SITE and TSYFWD_CACHE override the
output and cache directories.
"""

import os
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsyfwd import config, dashboard

if __name__ == "__main__":
    # pull the published history back in so the audit trail survives CI runs
    published = os.environ.get("HISTORY_URL", dashboard.SITE_URL + dashboard.HISTORY_NAME)
    dashboard.run(published_url=published)
    # static extras (robots.txt keeps the unlisted page out of search)
    for extra in (Path(__file__).resolve().parents[1] / "static").glob("*"):
        (config.SITE_DIR / extra.name).write_bytes(extra.read_bytes())
    if "--open" in sys.argv:
        webbrowser.open((config.SITE_DIR / "index.html").as_uri())
