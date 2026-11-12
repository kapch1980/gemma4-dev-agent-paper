"""Exp 43: replay our edit_file calls through the v6b heredoc editor (bundles/v6b/skills/editor/scripts/edit.py).

Usage: python3 scripts/editor_replay.py [--limit N] > results/editor-replay.log

For every trace with edit_file calls: unpack the task's repo snapshot (~/gemma4/data/snapshots/<task>.tgz, base
commit), walk the steps in order and keep the file state in step with the harness:
  - edit_file ok        -> feed (old_string, new_string) to the editor as one SEARCH/REPLACE block. If the editor
                           refuses, apply the harness result ourselves (exact, else unescaped) so later calls see
                           the same file. Records: would the editor have applied what edit_file applied?
  - edit_file not found -> try the editor (fuzzy matching), then restore the file. Records: rescued or not.
  - edit_file garbled   -> rebuild the intended old text from the split arguments (the parser cut it at colons
                           and commas, losing whitespace and the last 1-2 chars), find the unique file region whose whitespace-collapsed
                           text equals it, feed those exact lines + the recovered new text to the editor, restore.
  - write_file ok       -> write the content.
  - run_command that may change workspace files (sed -i, redirects outside /tmp, write_text, git checkout/apply,
    python heredocs) -> stop replaying this trace (we can't reproduce the file state); later calls are 'skipped'.
The editor gets the text exactly as a heredoc would carry it; when the harness needed its 'unescaped' strategy
(the model typed \\n inside JSON), the unescaped text is what the model would type in a heredoc, so we feed that.
"""
import argparse
import collections
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDITOR = os.path.join(ROOT, 'bundles/v6b/skills/editor/scripts/edit.py')
SNAP = os.path.expanduser('~/gemma4/data/snapshots')
MUTATING = re.compile(r"sed\s+-i|>\s*(?!/tmp|/dev/null|&)[\w./-]|write_text|open\([^)]*['\"][wa]|git\s+(checkout|apply|stash|restore)"
                      r"|\bpatch\b|\bmv\b|\bcp\b|\brm\b|python3?\s+-\s*<<|tee\s+(?!/tmp)")
MARKER = re.compile(r"(`|<\|\"\|>|\"|')?\s*,\s*old_string\s*:\s*")


def unescape(s):
    try:
        return s.encode('latin-1', 'backslashreplace').decode('unicode_escape')
    except Exception:
        return s


def body(raw):
    try:
        b = json.loads(raw)
    except Exception:
        return {}
    return b if isinstance(b, dict) else {}


def run_editor(ws, path, old, new):
    j = os.path.join(ws, '.git', 'swe_edit_journal')
    if os.path.exists(j):
        os.remove(j)
    block = f"<<<<<<< SEARCH\n{old.rstrip(chr(10))}\n=======\n{new.rstrip(chr(10))}\n>>>>>>> REPLACE\n"
    r = subprocess.run([sys.executable, EDITOR, path], input=block, cwd=ws, capture_output=True, text=True, timeout=60)
    out = r.stdout + r.stderr
    if r.returncode == 0:
        return ('no change' if out.startswith('NO CHANGE') else 'applied'), out
    for key, tag in (('not compile', 'refused: does not compile'), ('SyntaxError', 'refused: does not compile'),
                     ('not unique', 'refused: ambiguous'), ('occurs', 'refused: ambiguous'), ('ambiguous', 'refused: ambiguous'),
                     ('test', 'refused: test file'), ('not found', 'refused: not found'), ('No such file', 'refused: no file')):
        if key in out:
            return tag, out
    return 'refused: other', out


def snapshot(ws_path):
    with open(ws_path, 'rb') as fh:
        return fh.read()


def harness_apply(ws, path, old, new):
    p = os.path.join(ws, path)
    s = open(p, encoding='utf-8', errors='surrogateescape').read()
    for o, n in ((old, new), (unescape(old), unescape(new))):
        if o and s.count(o) >= 1:
            open(p, 'w', encoding='utf-8', errors='surrogateescape').write(s.replace(o, n, 1))
            return True
    return False


def squash(t):
    return re.sub(r'[\s,:]+', '', t)


def recover_garbled(args, text):
    """-> (old_lines_exact, new_text) or None."""
    new = args.get('new_string') or ''
    m = MARKER.search(new)
    extra = [(k, v) for k, v in args.items() if k not in ('filepath', 'new_string', 'allow_multiple')]
    if m:
        new_text, tail = new[:m.start()], new[m.end():]
    elif extra:
        new_text, tail = new, ''
    else:
        return None
    old_guess = tail + ''.join(f'{k}{v if isinstance(v, str) else ""}' for k, v in extra)
    old_guess = re.sub(r'(`|<\|"\|>)\s*$', '', old_guess)
    target = squash(unescape(old_guess))
    if len(target) < 15:
        return None
    lines = text.split('\n')
    sq = [squash(l) for l in lines]
    hits = []
    for i in range(len(lines)):
        acc = ''
        for j in range(i, min(i + 80, len(lines))):
            acc += sq[j]
            if len(acc) >= len(target):
                if acc.startswith(target) and len(acc) - len(target) <= 3:  # the parser drops the last 1-2 chars
                    hits.append((i, j))
                break
            if not target.startswith(acc):
                break
    if len(hits) != 1:
        return None
    i, j = hits[0]
    return '\n'.join(lines[i:j + 1]), unescape(new_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    traces = sorted(glob.glob(os.path.join(ROOT, 'results/*/results/traces/trace_*.json')))
    stats = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    done = 0
    for f in traces:
        steps = json.load(open(f))['steps']
        calls = [(tc, (s.get('observation') or {})) for s in steps for tc in (s.get('tool_calls') or [])]
        if not any(tc['function_name'] == 'edit_file' for tc, _ in calls):
            continue
        task = re.search(r'trace_(.+)\.json', f).group(1)
        tgz = os.path.join(SNAP, task + '.tgz')
        if not os.path.exists(tgz):
            stats['_'].update(['trace without snapshot'])
            continue
        tmp = tempfile.mkdtemp(prefix='replay_')
        with tarfile.open(tgz) as t:
            t.extractall(tmp, filter='data')
        ws = tmp
        stop = False
        for tc, o in calls:
            n, args = tc['function_name'], tc.get('arguments') or {}
            raw = o.get('content', '') if isinstance(o, dict) else str(o)
            b = body(raw)
            if n == 'run_command' and not stop and MUTATING.search(args.get('command') or ''):
                stop = True
            if n == 'write_file' and not stop and b.get('status') == 'ok':
                p = os.path.join(ws, args.get('filepath', ''))
                if os.path.realpath(p).startswith(os.path.realpath(ws)):
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    open(p, 'w').write(args.get('content') or '')
            if n != 'edit_file':
                continue
            garbled = 'mandatory input parameters' in raw
            em = b.get('error_message', '') or ''
            if 'BudgetExceeded' in (b.get('error_type') or '') or 'identical' in em:
                stats['_']['edit_file not executed (budget hit or old == new)'] += 1
                continue
            kind = 'garbled' if garbled else ('ok' if b.get('status') == 'ok' else
                   'not found' if 'not found' in b.get('error_message', '') else
                   'ambiguous' if 'occurs' in b.get('error_message', '') else 'other error')
            if stop:
                stats[kind]['skipped (state unknown)'] += 1
                continue
            path = args.get('filepath') or ''
            fp = os.path.join(ws, path)
            if not os.path.isfile(fp):
                stats[kind]['skipped (file missing)'] += 1
                continue
            before = snapshot(fp)
            if kind == 'garbled':
                rec = recover_garbled(args, before.decode('utf-8', 'surrogateescape'))
                if rec is None:
                    stats[kind]['not reconstructable'] += 1
                    continue
                old, new = rec
                stats[kind]['reconstructed'] += 1
            else:
                old, new = args.get('old_string') or '', args.get('new_string') or ''
                if kind == 'ok' and b.get('strategy') == 'unescaped':
                    old, new = unescape(old), unescape(new)
            res, out = run_editor(ws, path, old, new)
            if res != 'applied' and kind == 'ok' and b.get('strategy') != 'unescaped' and '\\' in old + new:
                res2, out2 = run_editor(ws, path, unescape(old), unescape(new))
                if res2 == 'applied':
                    res, out = 'applied (unescaped)', out2
            stats[kind][res] += 1
            if len(examples[(kind, res)]) < 2:
                examples[(kind, res)].append(f'{task} {path}: {out.strip()[:300]}')
            if kind == 'ok':
                if not res.startswith('applied'):
                    open(fp, 'wb').write(before)
                    if not harness_apply(ws, path, args.get('old_string') or '', args.get('new_string') or ''):
                        stop = True
            else:
                open(fp, 'wb').write(before)
        shutil.rmtree(tmp, ignore_errors=True)
        done += 1
        if a.limit and done >= a.limit:
            break
    print(f'traces replayed: {done}')
    for kind in ('ok', 'not found', 'ambiguous', 'other error', 'garbled', '_'):
        if stats[kind]:
            print(f'\n== edit_file {kind}: ' + ', '.join(f'{k} {v}' for k, v in stats[kind].most_common()))
    print('\n== examples')
    for (kind, res), ex in sorted(examples.items()):
        for e in ex:
            print(f'[{kind} -> {res}] {e}')


if __name__ == '__main__':
    main()
