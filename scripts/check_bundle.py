"""Compile a submission bundle with the grader's own adk_submission/swegemma (put the wheel dir first on sys.path).

Usage: ~/gemma4/.venv/bin/python scripts/check_bundle.py WHEEL_DIR bundles/v6a [bundles/b024 ...]
Prints: tools the agent gets, the generate_content_config the grader builds, the budget, and whether the harness
would add the "Code Intelligence Tools" section to the task prompt. (Approach after Emanuele Miranda's
official_check.py, Apache 2.0, github.com/emamira02/gemma4-swe-agent.)
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from adk_submission import ModelRegistry, compile_submission, validate_directory  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from swegemma.config import build_submission_limits  # noqa: E402

MODEL = 'gemma-4-31b-it-qat-w4a16-ct'


def stand_ins():
    def run_command(command: str) -> str:
        """Run a shell command."""
        return '{}'

    def submit_patch() -> str:
        """Submit."""
        return '{}'

    def get_status() -> str:
        """Status."""
        return '{}'

    def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read."""
        return '{}'

    def edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str:
        """Edit."""
        return '{}'

    def write_file(filepath: str, content: str) -> str:
        """Write."""
        return '{}'

    def get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str:
        """Graph."""
        return '{}'

    def search_similar_code(query: str, k: int = 10) -> str:
        """Graph."""
        return '{}'

    def get_code_subgraph(nodes: list[str]) -> str:
        """Graph."""
        return '{}'

    fs = (run_command, submit_patch, get_status, read_file, edit_file, write_file, get_code_neighbors,
          search_similar_code, get_code_subgraph)
    return {f.__name__: f for f in fs}


async def tool_names(agent):
    out = []
    for t in agent.tools:
        out += [x.name for x in await t.get_tools()] if hasattr(t, 'get_tools') else [getattr(t, 'name', str(t))]
    return out


for b in sys.argv[2:]:
    bundle = Path(b).resolve()
    limits, gen = build_submission_limits()
    validate_directory(str(bundle), limits=limits)
    models = ModelRegistry()
    models.register(MODEL, LiteLlm(model=f'openai/{MODEL}', api_base='http://127.0.0.1:9/v1', api_key='EMPTY'))
    agent = compile_submission(submission_dir=bundle, tool_registry=stand_ins(), model_registry=models,
                               limits=limits, generation_constraints=gen, script_timeout=240)
    tools = asyncio.run(tool_names(agent))
    gc = agent.generate_content_config
    graph = any(t in tools for t in ('search_similar_code', 'get_code_neighbors', 'get_code_subgraph'))
    print(f'== {b}: OK')
    print('  tools:', tools)
    print('  generate_content_config:', gc.model_dump_json(exclude_none=True) if gc else None)
    print('  eval_config:', (bundle / 'eval_config.yaml').read_text().split('\n', 1)[1].replace('\n', ' '))
    print('  prompt words:', len((agent.instruction or '').split()), '| braces in prompt:',
          '{' in (agent.instruction or '') or '}' in (agent.instruction or ''))
    print('  harness adds "Code Intelligence Tools" section:', graph)
