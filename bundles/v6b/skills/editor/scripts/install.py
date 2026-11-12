#!/usr/bin/env python3
"""Install the SEARCH/REPLACE editor into <workspace>/.git/swe/edit.py so run_command can call it.

install_editor() and find_root() are copied from emamira02/gemma4-swe-agent (Apache-2.0, commit 9a53cf5,
agent/skills/swe/scripts/finalize.py); see ../LICENSE.txt. Unchanged apart from this header.
"""
from __future__ import annotations

import os

TOOLS_DIR = os.environ.get("SWE_TOOLS_DIR")  # tests only; the default is the workspace's .git directory
EDITOR_CMD = "/workspace/.git/swe/edit.py"


def install_editor(root: str) -> str:
    """Copy the sibling edit.py (materialized next to this script by run_skill_script) to <workspace>/.git/swe/,
    where run_command can call it after this script's temporary directory is gone. Commands spell it
    /workspace/.git/swe/edit.py: the harness rewrites /workspace and /tmp inside run_command strings to the task
    sandbox but not inside this Python process, so a /tmp install is invisible to run_command (S6 dev panel: the
    editor was not found in most tasks). Files under .git never enter the patch. Returns the status line to print."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "edit.py"), "rb") as fh:
            data = fh.read()
    except OSError as e:
        return f"EDITOR NOT INSTALLED ({type(e).__name__}); run this script once more."
    d = TOOLS_DIR or os.path.join(root, ".git", "swe")
    dst = os.path.join(d, "edit.py")
    try:
        os.makedirs(d, exist_ok=True)
        old = None
        if os.path.isfile(dst):
            with open(dst, "rb") as fh:
                old = fh.read()
        if old != data:
            tmp = f"{dst}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dst)
    except OSError as e:
        return f"EDITOR NOT INSTALLED ({type(e).__name__}: {e}); run this script once more."
    in_workspace = not TOOLS_DIR and os.path.basename(os.path.normpath(root)) == "workspace"
    return f"EDITOR READY: python3 {EDITOR_CMD if in_workspace else dst.replace(os.sep, '/')}"



def find_root() -> str:
    cands = [os.environ.get("SWE_WORKSPACE"), os.environ.get("PWD"),
             *os.environ.get("PYTHONPATH", "").split(os.pathsep), "/workspace"]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, ".git")):
            return os.path.realpath(c)
    return "/workspace"


if __name__ == "__main__":
    print(install_editor(find_root()))
