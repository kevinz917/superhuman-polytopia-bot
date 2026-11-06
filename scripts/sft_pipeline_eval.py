"""One evaluation arm of an SFT-pipeline run (Modal only): ``benchmark_expert_sft`` with the
protocol's prompt renderer.

  python scripts/sft_pipeline_eval.py <run>/eval <arm> [--preflight-only]

The benchmark hard-codes the readable renderer and grammar; for any other registered
renderer this wrapper rebinds those module globals before the run (no file edits). The
eval protocol records the renderer, so a mismatch with the training data is impossible.
"""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_expert_sft as bench  # noqa: E402
import benchmark_qwen3_instruct as instruct  # noqa: E402
from polytopia_bot.learning.sft_pipeline import resolve_renderer  # noqa: E402


def main(root, arm, preflight_only):
    p = json.loads((root/'protocol.json').read_text())
    spec = p['prompt_renderer']
    if spec['name'] != 'readable-v1':
        fns = resolve_renderer(spec)
        bench.render_readable_decision = fns['render']
        instruct.render_readable_decision = fns['render']
        if fns['schema'] is None:
            raise ValueError('renderer has no evaluation grammar (schema)')
        instruct.action_schema = fns['schema']
    asyncio.run(bench.gameplay(root, arm, preflight_only=preflight_only))


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], '--preflight-only' in sys.argv)
