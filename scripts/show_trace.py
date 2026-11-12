"""Print one Gemma trace as readable text: one block per step (what Gemma said, which tool it called with what
arguments, what came back). For trace analysis by hand.

Usage:
  python3 scripts/show_trace.py RUN TASK              # e.g. eval10-v5 fastapi_13713
  python3 scripts/show_trace.py RUN TASK --answer     # also print the real fix (answer key: for us only)
  python3 scripts/show_trace.py --runs TASK           # list every run that has this task, solved or not
  python3 scripts/show_trace.py RUN TASK > /tmp/t.txt # save it to read in an editor

Long texts are cut to --width characters (default 600); use --width 0 for everything.
"""
import argparse, csv, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cut(s, w):
    s = s if isinstance(s, str) else json.dumps(s)
    return s if not w or len(s) <= w else s[:w] + f' ...[{len(s) - w} more chars]'


def outcome(run, task):
    p = os.path.join(ROOT, 'results', run, 'eval_results.csv')
    if not os.path.exists(p):
        return None
    for r in csv.DictReader(open(p)):
        if r['id'] == task:
            return r
    return None


def list_runs(task):
    for f in sorted(glob.glob(os.path.join(ROOT, f'results/*/results/traces/trace_{task}.json'))):
        run = f.split('/results/')[-2].split('/')[-1]
        r = outcome(run, task)
        res = '?' if r is None else ('SOLVED' if r['resolved'] == 'True' else 'failed')
        extra = '' if r is None else f"  calls={r['tool_calls']}  time={r['duration_s']}s  patch_chars={r['patch_chars']}"
        print(f'{run:28s} {res}{extra}')


def show(run, task, w, answer):
    f = os.path.join(ROOT, 'results', run, 'results/traces', f'trace_{task}.json')
    d = json.load(open(f))
    r = outcome(run, task)
    print(f'=== {task} in run {run}')
    if r:
        print(f"=== result: {'SOLVED' if r['resolved'] == 'True' else 'FAILED'}, {r['tool_calls']} tool calls, "
              f"{r['duration_s']} s, patch {r['patch_chars']} chars")
    n = 0
    for s in d['steps']:
        if s.get('source') == 'user':
            print('\n--- TASK GIVEN TO GEMMA ---\n' + cut(s.get('message', ''), w or 0))
            continue
        if s.get('source') != 'agent':
            continue
        msg = (s.get('message') or '').strip()
        for tc in s.get('tool_calls') or [{}]:
            n += 1
            print(f'\n##### step {n}')
            if msg:
                print('GEMMA SAYS: ' + cut(msg, w))
                msg = ''
            if tc:
                print(f"TOOL: {tc.get('function_name')}  ARGS: {cut(tc.get('arguments'), w)}")
        obs = s.get('observation')
        if obs:
            c = obs.get('content') if isinstance(obs, dict) else obs
            print('RESULT: ' + cut(c, w))
    print(f"\n=== total steps {d.get('final_metrics', {}).get('total_steps')}")
    if answer:
        for l in open(os.path.join(ROOT, 'data/tasks.jsonl')):
            t = json.loads(l)
            if t['instance_id'] == task:
                print('\n=== ANSWER KEY (real fix) - for understanding only, never put any of it in a prompt ===')
                print(t['patch'])
                print('=== HIDDEN TESTS ===')
                print(t['test_patch'])


ap = argparse.ArgumentParser()
ap.add_argument('args', nargs='+')
ap.add_argument('--runs', action='store_true')
ap.add_argument('--answer', action='store_true')
ap.add_argument('--width', type=int, default=600)
a = ap.parse_args()
if a.runs:
    list_runs(a.args[0])
else:
    show(a.args[0], a.args[1], a.width, a.answer)
