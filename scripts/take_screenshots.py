"""Seed a demo profile, run the server, and capture screenshots of the UI.

Output: docs/screenshots/*.png. Re-run after UI changes to refresh.

Requirements: chromium / chromium-browser on PATH.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "screenshots"
DB_PATH = "/tmp/a5c_engram_screenshots.db"
PORT = 8765


def wait_for_port(host: str, port: int, timeout: float = 15.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(0.2)
    raise TimeoutError(f"{host}:{port} not listening after {timeout}s")


def seed() -> None:
    # Fresh DB so the screenshot is deterministic.
    for p in (DB_PATH, f"{DB_PATH}-wal", f"{DB_PATH}-shm"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    os.environ["A5C_ENGRAM_EMBEDDER"] = "fake"  # avoid model download for screenshots
    sys.path.insert(0, str(ROOT / "src"))
    from a5c_engram import Profile

    p = Profile.open("atlas", db_path=DB_PATH)

    p.ingest(
        [
            {"role": "user", "content": "Quick standup: rebuilt the auth service last sprint."},
            {"role": "user", "content": "Token TTL is 24h, refresh extends by 24h."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "Always run migrations during the Sunday window."},
        ],
        session_id="standup-2026-05-22",
    )

    # Supersession chain.
    p.remember("project uses GraphQL", type="fact", topic="api_style")
    p.remember("project uses REST", type="fact", topic="api_style")
    p.remember("project uses gRPC", type="fact", topic="api_style")

    # A few other memories.
    p.remember("Never deploy on Fridays.", type="instruction", topic="deploy_policy")
    p.remember("Run migrations on the Sunday window only.", type="instruction", topic="migration_window")
    p.remember("Deployed v2.1 to production on 2026-05-21.", type="event")
    p.remember("Review PR #432 by Thursday.", type="task")
    p.remember("Postgres uses MVCC for read isolation.", type="fact", topic="database_concurrency")

    # A second profile so the index page has more than one row.
    nova = Profile.open("nova", db_path=DB_PATH)
    nova.remember("primary metric is conversion rate.", type="fact", topic="kpi")
    nova.remember("Q3 OKR: lift conversion 12%.", type="fact", topic="okr_q3")
    nova.remember("Standup is Tuesday 10am.", type="instruction", topic="meeting_schedule")


def snap(url: str, out: Path, height: int = 900) -> None:
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if chromium is None:
        raise RuntimeError("chromium not found on PATH")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        chromium,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size=1280,{height}",
        "--default-background-color=ffffff",
        f"--screenshot={out}",
        url,
    ]
    print(f"  → {out.relative_to(ROOT)}")
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    print("[1] seeding demo profile…")
    seed()

    print("[2] starting server…")
    env = os.environ.copy()
    env["A5C_ENGRAM_DB"] = DB_PATH
    env["A5C_ENGRAM_PORT"] = str(PORT)
    env["A5C_ENGRAM_EMBEDDER"] = "fake"
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "a5c_engram.server"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    try:
        wait_for_port("127.0.0.1", PORT)
        time.sleep(0.5)  # let any first-paint settle

        base = f"http://127.0.0.1:{PORT}"
        print("[3] capturing pages…")
        snap(f"{base}/ui", OUT_DIR / "profiles.png", height=600)
        snap(
            f"{base}/ui/p/atlas/recall?q=what+api+do+we+use%3F",
            OUT_DIR / "recall.png",
            height=1100,
        )
        # Find a fact_key memory to deep-link to.
        from a5c_engram import Profile

        p = Profile.open("atlas", db_path=DB_PATH)
        latest = p.storage.latest_for_factkey("atlas", "api_style")
        if latest is not None:
            snap(
                f"{base}/ui/p/atlas/m/{latest.id}",
                OUT_DIR / "memory_detail.png",
                height=700,
            )
        snap(f"{base}/ui/p/atlas", OUT_DIR / "memories.png", height=900)

        print("[4] done.")
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
