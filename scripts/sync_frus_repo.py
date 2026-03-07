from __future__ import annotations

import os
import sys

import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FRUS_GIT_URL, FRUS_REPO_DIR


SPARSE_VOLUME_PATTERNS = [
    "volumes/frus196*",
    "volumes/frus197*",
    "volumes/frus198*",
    "volumes/frus199*",
    "volumes/frus20*",
]


def run(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def configure_sparse_checkout() -> None:
    run(["git", "-C", str(FRUS_REPO_DIR), "config", "core.sparseCheckout", "true"])
    run(["git", "-C", str(FRUS_REPO_DIR), "config", "core.sparseCheckoutCone", "false"])

    sparse_file = FRUS_REPO_DIR / ".git" / "info" / "sparse-checkout"
    sparse_file.parent.mkdir(parents=True, exist_ok=True)
    sparse_file.write_text("\n".join(SPARSE_VOLUME_PATTERNS) + "\n", encoding="utf-8")

    run(["git", "-C", str(FRUS_REPO_DIR), "read-tree", "-mu", "HEAD"])


def sync_frus_repo() -> None:
    FRUS_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not FRUS_REPO_DIR.exists():
        print(f"Cloning FRUS repo into {FRUS_REPO_DIR}...")
        run(["git", "clone", FRUS_GIT_URL, str(FRUS_REPO_DIR)])
        configure_sparse_checkout()
    else:
        print(f"Pulling latest FRUS changes in {FRUS_REPO_DIR}...")
        configure_sparse_checkout()
        run(["git", "-C", str(FRUS_REPO_DIR), "fetch", "origin"])
        run(["git", "-C", str(FRUS_REPO_DIR), "pull", "--ff-only", "origin"])


if __name__ == "__main__":
    sync_frus_repo()
