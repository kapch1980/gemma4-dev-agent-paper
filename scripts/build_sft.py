"""Turn solved agent traces (ATIF v1.7) into chat-format SFT examples for LoRA training.

Usage: python3 scripts/build_sft.py OUT.jsonl SOLVED_IDS.txt TRACE_DIR [TRACE_DIR ...]

Each output line is one solved task:
  {"id", "repo", "messages": [system, user, assistant(+tool_calls), tool, ..., assistant], "meta": {...}}
Assistant "content" holds the model's reasoning text logged before each tool call.
Tool schemas are not stored in traces; the training notebook adds them from the harness.
"""
import json
import re
import sys
from pathlib import Path

# Hinted runs (notebooks/push-hinted-1) append a location hint after this marker; it is cut from the user turn.
HINT_MARKER = '\n\n---\nLOCATION HINT:'
# A trace that talks about the hint would teach the model to expect one, so it is flagged and skipped.
HINT_TALK = re.compile(r'location hint|\bthe hint\b|\bhinted\b|hint (says|points|suggests|mentions)', re.I)


def load_ids(path):
    return {line.strip() for line in open(path) if line.strip()}


def convert(trace, task_id, require_final=True):
    msgs, meta = [], {'tool_calls': 0, 'tool_errors': 0, 'repeat_calls': 0, 'chars': 0}
    prev_call = None
    for step in trace['steps']:
        event = (step.get('extra') or {}).get('event_type')
        src, text = step.get('source'), step.get('message') or ''
        if src == 'system':
            msgs.append({'role': 'system', 'content': text})
        elif src == 'user':
            meta['hinted'] = meta.get('hinted', False) or HINT_MARKER in text
            msgs.append({'role': 'user', 'content': text.split(HINT_MARKER)[0]})
        elif src == 'agent':
            calls = step.get('tool_calls') or []
            msg = {'role': 'assistant', 'content': text}
            if calls:
                msg['tool_calls'] = [{
                    'id': c['tool_call_id'], 'type': 'function',
                    'function': {'name': c['function_name'], 'arguments': json.dumps(c['arguments'], ensure_ascii=False)},
                } for c in calls]
            msgs.append(msg)
            obs = step.get('observation')
            if calls and obs is not None:
                # One observation per step; multi-call steps are rare (1 in ~2000) and get the shared result.
                for c in calls:
                    msgs.append({'role': 'tool', 'tool_call_id': c['tool_call_id'], 'name': c['function_name'], 'content': obs.get('content', '')})
                meta['tool_calls'] += len(calls)
                try:
                    status = json.loads(obs.get('content', '{}')).get('status')
                except (ValueError, AttributeError):
                    status = None
                if status not in (None, 'ok'):
                    meta['tool_errors'] += 1
                key = json.dumps([(c['function_name'], c['arguments']) for c in calls], sort_keys=True)
                meta['repeat_calls'] += key == prev_call
                prev_call = key
            elif calls:
                # Call with no recorded result (budget ran out): drop the dangling call, keep the text.
                del msg['tool_calls']
    meta['chars'] = sum(len(m['content']) + len(json.dumps(m.get('tool_calls', ''))) for m in msgs)
    meta['approx_tokens'] = meta['chars'] // 3  # rough; real count comes from the Gemma tokenizer
    meta['hint_talk'] = meta.get('hinted', False) and any(m['role'] == 'assistant' and HINT_TALK.search(m['content']) for m in msgs)
    roles = [m['role'] for m in msgs]
    assert roles[:2] == ['system', 'user'], (task_id, roles[:3])
    assert not require_final or msgs[-1]['role'] == 'assistant', (task_id, roles[-3:])
    return msgs, meta


def main(out, ids_path, *trace_dirs):
    ids = load_ids(ids_path)
    held_out = load_ids(Path(__file__).resolve().parent.parent / 'results' / 'eval_set_30.txt')
    assert not ids & held_out, f'held-out tasks in training ids: {sorted(ids & held_out)}'
    found = {}
    for d in trace_dirs:
        for f in Path(d).glob('trace_*.json'):
            tid = f.stem.removeprefix('trace_')
            if tid in ids:
                found[tid] = f
    missing = ids - found.keys()
    assert not missing, f'no trace for {sorted(missing)}'
    with open(out, 'w') as fh:
        for tid in sorted(found):
            msgs, meta = convert(json.load(open(found[tid])), tid)
            if meta['hint_talk']:
                print(f'{tid:16s} SKIPPED: the reasoning mentions the hint')
                continue
            repo = 'fastapi/fastapi' if tid.startswith('fastapi') else 'Textualize/rich'
            fh.write(json.dumps({'id': tid, 'repo': repo, 'messages': msgs, 'meta': meta}, ensure_ascii=False) + '\n')
            print(f"{tid:16s} calls={meta['tool_calls']:3d} errors={meta['tool_errors']:2d} repeats={meta['repeat_calls']} ~tokens={meta['approx_tokens']:6d}")
    print(f'wrote examples to {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
