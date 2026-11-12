---
name: editor
description: Installs a SEARCH/REPLACE file editor that is called from run_command with a quoted heredoc.
---
# editor skill

Run scripts/install.py once with run_skill_script (skill_name editor, file_path scripts/install.py). It prints
EDITOR READY. Then change files with run_command:

python3 /workspace/.git/swe/edit.py path/to/module.py <<'EOF'
<<<<<<< SEARCH
exact current lines
=======
replacement lines
>>>>>>> REPLACE
EOF

The editor is edit.py from emamira02/gemma4-swe-agent (Apache-2.0, commit 9a53cf5), unchanged; see LICENSE.txt.
