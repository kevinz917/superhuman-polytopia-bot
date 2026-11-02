"""Check the local coding-plan catalog; --live also tests bounded tool continuations.

Never writes raw responses, credentials, OAuth files, or benchmark submissions.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from polytopia_bot.skirmish.inference import endpoint
from polytopia_bot.skirmish.providers import provider


def catalog():
    import urllib.request
    route = endpoint(dict(provider='cliproxyapi', model='gpt-6-luna'))
    req = urllib.request.Request(route.base_url+'/models', headers={'Authorization': 'Bearer '+route.credential()})
    with urllib.request.urlopen(req, timeout=10) as response:
        available = {m['id'] for m in json.load(response)['data']}
    roster = json.loads((Path(__file__).resolve().parents[1]/'configs/skirmish/cli-proxy-models.json').read_text())
    rows = [{**item, 'available': endpoint(item).model in available} for item in roster['models']]
    if not all(r['available'] for r in rows):
        raise RuntimeError('Requested proxy models missing: '+', '.join(r['label'] for r in rows if not r['available']))
    return rows


def native_check(name, model):
    cfg = dict(provider=name, model=model, parameters=dict(max_tokens=1024))
    if name in ('chat', 'responses'):
        cfg['parameters']['reasoning'] = dict(effort='low')
    api = provider(cfg, 'Transport test only. First call check exactly once with nonce proxy-ok. '
                   'After receiving its result, reply exactly PROXY_OK without another tool call.',
                   [dict(name='check', description='Echo a nonce.', parameters=dict(type='object',
                        properties=dict(nonce=dict(type='string')), required=['nonce'], additionalProperties=False))])
    turn = api.next('Start the check.', 60)
    assert len(turn['calls']) == 1 and turn['calls'][0]['name'] == 'check', 'Expected check tool'
    assert json.loads(turn['calls'][0]['arguments']) == dict(nonce='proxy-ok'), 'Unexpected tool arguments'
    api.result(turn['calls'][0], dict(ok=True, nonce='proxy-ok'))
    final = api.next(None, 60)
    assert not final['calls'] and 'PROXY_OK' in json.dumps(final['raw']), 'Continuation failed'
    return dict(protocol=name, **api.endpoint.receipt(), tool_roundtrip=True,
                calls=2, usage=[turn['usage'], final['usage']], api_charge_usd=None)


def mini_check(model, config=None):
    from polytopia_bot.skirmish.mini_worker import coding_model
    from minisweagent.exceptions import FormatError
    api = coding_model(config or dict(model=model, parameters=dict(max_tokens=4096)))
    messages = [api.format_message(role='system', content='You are a coding agent. Use bash to run commands. '
                'First run printf PROXY_OK, then after its result run printf CONTINUATION_OK. '
                'Every response must contain a bash tool call.'),
                api.format_message(role='user', content='Start the tool check.')]
    repairs = 0
    calls = 0
    def query():
        nonlocal repairs, calls
        for attempt in range(3):
            calls += 1
            try:
                return api.query(messages)
            except FormatError as error:
                repairs += 1
                response = error.messages[0].get('extra', {}).get('response', {})
                print(json.dumps(dict(mini_format_repair=repairs,
                    model=model, finish_reason=(response.get('choices') or [{}])[0].get('finish_reason'),
                    error=error.messages[0]['content'])),file=sys.stderr,flush=True)
                messages.extend(error.messages)
        raise RuntimeError('Mini transport check exceeded format-repair allowance')
    turn = query()
    assert turn['extra']['actions'], 'Expected at least one parsed bash action'
    messages.append(turn)
    messages.extend(api.format_observation_messages(turn, [dict(output='PROXY_OK', returncode=0, exception_info='')
        for action in turn['extra']['actions']]))
    final = query()
    assert final['extra']['actions'], 'Expected parsed bash action after tool result'
    route = api.serialize()['info']['inference']
    costs = [(t['extra']['response'].get('usage') or {}).get('cost') for t in [turn,final]]
    known_cost = sum(costs) if all(isinstance(c,(int,float)) for c in costs) else None
    return dict(protocol='mini-swe-agent', **route,tool_roundtrip=True,calls=calls,
                format_repairs=repairs,
                nonce_instruction_followed=(any('PROXY_OK' in a['command'] for a in turn['extra']['actions']) and
                    any('CONTINUATION_OK' in a['command'] for a in final['extra']['actions'])),
                api_charge_usd=known_cost if route['backend']=='openrouter' else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Dispatch small coding-plan transport checks, not benchmark runs')
    parser.add_argument('--only', choices=['chat', 'responses', 'anthropic', 'fable', 'mini'],
                        action='append', help='Limit live checks to selected protocols/models')
    args = parser.parse_args()
    result = dict(catalog=catalog())
    if args.live:
        result['checks'] = []
        for label, name, model in [
            ('chat', 'chat', 'openai/gpt-6-luna'), ('responses', 'responses', 'openai/gpt-6-luna'),
            ('anthropic', 'anthropic', 'anthropic/claude-haiku-4-5-20251001'),
            ('fable', 'anthropic', 'anthropic/claude-fable-5-1'),
            ('mini', 'mini', 'anthropic/claude-fable-5-1')]:
            if args.only and label not in args.only:
                continue
            check = mini_check(model) if name == 'mini' else native_check(name, model)
            result['checks'].append(check)
            print(json.dumps(dict(check=label, passed=True)), file=sys.stderr, flush=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
