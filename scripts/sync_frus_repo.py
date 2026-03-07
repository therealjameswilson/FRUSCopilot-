from __future__ import annotations

import subprocess

from config import FRUS_GIT_URL, FRUS_REPO_DIR


def run(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def sync_frus_repo() -> None:
    FRUS_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not FRUS_REPO_DIR.exists():
        print(f"Cloning FRUS repo into {FRUS_REPO_DIR}...")
        run(["git", "clone", FRUS_GIT_URL, str(FRUS_REPO_DIR)])
    else:
        print(f"Pulling latest FRUS changes in {FRUS_REPO_DIR}...")
        run(["git", "-C", str(FRUS_REPO_DIR), "fetch", "origin"])
        run(["git", "-C", str(FRUS_REPO_DIR), "pull", "--ff-only", "origin"])


if __name__ == "__main__":
    sync_frus_repo()
