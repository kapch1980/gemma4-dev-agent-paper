"""Count tool-mechanics failures in agent traces (Exp 37/38).

Usage: python3 scripts/mechanics_audit.py results/eval10-v4 results/eval10-v5 ...
Per run: tasks, calls, garbled edit_file ("mandatory input parameters are not present") and how many were resent
verbatim, write_file failures, read_file calls cut at 150 lines or without end_line, grep -r without --include,
python -c SyntaxErrors, identical repeated calls. v6b (Exp 42): editor install calls, heredoc editor calls applied /
not applied / editor missing.
"""
import collections
import glob
import json
import re
import sys


def audit(run):
    c = collections.Counter()
    for f in glob.glob(f'{run}/results/traces/trace_*.json'):
        c['tasks'] += 1
        seen, prev_bad = collections.Counter(), None
        for s in json.load(open(f))['steps']:
            o = s.get('observation') or {}
            raw = o.get('content', '') if isinstance(o, dict) else str(o)
            try:
                body = json.loads(raw)
            except Exception:
                body = {}
            body = body if isinstance(body, dict) else {}
            for tc in s.get('tool_calls') or []:
                n, a = tc['function_name'], tc.get('arguments') or {}
                key = json.dumps([n, a], sort_keys=True)
                c['calls'] += 1
                seen[key] += 1
                c['identical repeats'] += seen[key] > 1
                if n == 'edit_file':
                    c['edit_file'] += 1
                    if 'mandatory input parameters' in raw:
                        c['edit_file garbled'] += 1
                        c['  garbled resent verbatim'] += prev_bad == key
                        prev_bad = key
                if n == 'write_file' and body.get('status') == 'error':
                    c['write_file error'] += 1
                if n == 'read_file':
                    c['read_file'] += 1
                    c['  read cut at 150 lines'] += bool(body.get('is_truncated'))
                    c['  read without end_line'] += a.get('end_line') is None
                if n == 'run_skill_script':
                    c['run_skill_script'] += 1
                    c['  EDITOR READY'] += 'EDITOR READY' in raw
                if n == 'run_command':
                    cmd = a.get('command') or ''
                    if 'swe/edit.py' in cmd:
                        c['editor calls'] += 1
                        c['  editor APPLIED'] += 'APPLIED to' in raw and 'NOT APPLIED' not in raw
                        c['  editor NOT APPLIED'] += 'NOT APPLIED' in raw
                        c['  editor missing'] += 'No such file' in raw and 'edit.py' in raw
                    c['heredoc scripts'] += '<<' in cmd
                    c['grep -r without --include'] += bool(re.search(r'grep -\w*r', cmd)) and '--include' not in cmd
                    c['python -c SyntaxError'] += bool(re.search(r'python3? -c', cmd)) and 'SyntaxError' in raw
    return c


runs = sys.argv[1:]
res = {r: audit(r) for r in runs}
keys = list(dict.fromkeys(k for r in runs for k in res[r]))
print(f"{'':28s}" + ''.join(f'{r.split("/")[-1]:>20s}' for r in runs))
for k in keys:
    print(f'{k:28s}' + ''.join(f'{res[r][k]:20d}' for r in runs))
