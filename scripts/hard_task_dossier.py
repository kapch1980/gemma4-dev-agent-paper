"""Build one plain-text dossier per unsolved task: the issue, the real fix and hidden tests (answer key, for us
only, never shown to Gemma), and a compact summary of each failed attempt (calls, files read and edited, whether it
touched the right file, repeated calls, last git diff, final message). Used for the hard-task failure buckets.

Usage: python3 scripts/hard_task_dossier.py OUT_DIR task_id [task_id ...]
"""
import collections, csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ['unleashed-1', 'unleashed-2', 'unleashed-3', 'shuchita-unleashed-4', 'shuchita-unleashed-5',
        'shuchita-unleashed-6', 'hinted-1']
TASKS = {json.loads(l)['instance_id']: json.loads(l) for l in open(ROOT / 'data/tasks.jsonl')}


def files_in(patch):
    return sorted(set(re.findall(r'^\+\+\+ b/(\S+)', patch, re.M)))


def obs_text(s):
    o = s.get('observation')
    if not o:
        return ''
    try:
        c = json.loads(o['content']) if isinstance(o, dict) and isinstance(o.get('content'), str) else o
    except Exception:
        c = o
    if isinstance(c, dict):
        return c.get('stdout') or c.get('content') or c.get('error') or json.dumps(c)
    return str(c)


def attempt(run, tid, gold_files):
    f = ROOT / f'results/{run}/results/traces/trace_{tid}.json'
    if not f.exists():
        return f'--- {run}: no trace\n'
    row = {r['id']: r for r in csv.DictReader(open(ROOT / f'results/{run}/eval_results.csv'))}.get(tid, {})
    t = json.load(open(f))
    calls, edited, read, last_diff, final, sigs = [], [], [], '', '', collections.Counter()
    for s in t['steps']:
        if s.get('source') != 'agent':
            continue
        if (s.get('extra') or {}).get('event_type') == 'final' and isinstance(s.get('message'), str):
            final = s['message']
        for c in s.get('tool_calls') or []:
            name, a = c['function_name'], c.get('arguments') or {}
            sig = name + ':' + json.dumps(a, sort_keys=True)[:300]
            sigs[sig] += 1
            short = a.get('filepath') or a.get('command') or a.get('query') or ''
            calls.append(f"{name}({str(short)[:90]})")
            if name in ('edit_file', 'write_file'):
                edited.append(a.get('filepath', '?'))
            elif name == 'read_file':
                read.append(a.get('filepath', '?'))
            elif name == 'run_command' and re.search(r"sed -i|> *\S+\.py|open\(.+['\"]w['\"]", str(a.get('command', ''))):
                edited.append('run_command:' + str(a.get('command'))[:80])
            if name == 'run_command' and 'git diff' in str(a.get('command', '')):
                d = obs_text(s)
                if 'diff --git' in d:
                    last_diff = d
    reps = [(k[:120], n) for k, n in sigs.items() if n >= 3]
    ed_files = sorted(set(e for e in edited if not e.startswith('run_command')))
    test_edit = [e for e in ed_files if re.search(r'(^|/)tests?/|test_|conftest', e)]
    out = [f"--- {run}: resolved={row.get('resolved')} tool_calls={row.get('tool_calls')} "
           f"duration={row.get('duration_s')}s patch_chars={row.get('patch_chars')}",
           f"edited files: {ed_files}  (gold files: {gold_files}; touched gold: {bool(set(ed_files) & set(gold_files))})",
           f"edited test files: {test_edit}" if test_edit else '',
           f"files read: {collections.Counter(read).most_common(8)}",
           f"calls repeated >=3x identically: {reps}" if reps else '',
           f"call sequence: {' > '.join(calls)[:2500]}",
           f"last git diff seen (truncated):\n{last_diff[:3500]}" if last_diff else 'last git diff seen: none',
           f"final message: {final[:600]}"]
    return '\n'.join(x for x in out if x) + '\n'


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for tid in sys.argv[2:]:
        task = TASKS[tid]
        gold = files_in(task['patch'])
        tests = re.findall(r'^\+\s*(?:async )?def (test_\w+)', task['test_patch'], re.M)
        txt = [f"===== {tid} ({task['repo']})", '## ISSUE (what Gemma sees)', task['problem_statement'][:5000],
               '## ANSWER KEY (for analysis only; never shown to Gemma)', f'gold files: {gold}',
               task['patch'][:5000], f'hidden tests added/changed: {tests}', task['test_patch'][:3500],
               '## FAILED ATTEMPTS (unleashed runs: 100 calls / 30 min; hinted-1 had a hint in the prompt)']
        txt += [attempt(r, tid, gold) for r in RUNS]
        (out / f'{tid}.txt').write_text('\n'.join(txt))
        print(tid, len('\n'.join(txt)), 'chars')


if __name__ == '__main__':
    main()
