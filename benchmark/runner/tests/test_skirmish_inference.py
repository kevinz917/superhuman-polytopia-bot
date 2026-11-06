"""Routing, credential isolation and real HTTP protocol continuation (no model calls)."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from polytopia_bot.skirmish.inference import PROXY_BASE, endpoint, parameters, proxy_model
from polytopia_bot.skirmish.providers import provider


@pytest.fixture(autouse=True)
def clean_routing_environment(monkeypatch):
    for name in ('MODAL_TASK_ID', 'CLIPROXY_BASE_URL', 'CLIPROXY_API_KEY', 'CLIPROXY_KEY_FILE'):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize('name,model', [
    ('openai', 'openai/gpt-6-luna'), ('responses', 'gpt-6-astra'),
    ('chat', 'gpt-6.1-sol'), ('anthropic', 'anthropic/claude-opus-5.5'),
    ('auto', 'claude-fable-5-1'), ('cliproxyapi', 'anthropic/claude-fable-5-1'),
])
def test_local_native_and_auto_defaults_use_proxy(name, model):
    route = endpoint(dict(provider=name, model=model))
    assert route.backend == 'cliproxyapi' and route.base_url == PROXY_BASE
    assert '/' not in route.model


def test_explicit_paid_routes_remain_explicit(monkeypatch):
    monkeypatch.setenv('CLIPROXY_BASE_URL', 'http://localhost:1234/v1')
    route = endpoint(dict(provider='openrouter', model='openai/gpt-6-luna'))
    assert route.backend == 'openrouter' and route.base_url == 'https://openrouter.ai/api/v1'
    assert route.model == 'openai/gpt-6-luna'
    route = endpoint(dict(provider='openai', model='gpt-6-luna', transport='direct'))
    assert route.backend == 'direct' and route.base_url == 'https://api.openai.com/v1'
    with pytest.raises(ValueError):
        endpoint(dict(provider='openrouter', model='gpt-6-luna', transport='cliproxyapi'))


def test_proxy_key_never_uses_paid_api_credentials_or_oauth(monkeypatch, tmp_path):
    keyfile = tmp_path/'client-keys.json'
    keyfile.write_text(json.dumps(dict(api_key='proxy-client-secret')))
    monkeypatch.setenv('CLIPROXY_KEY_FILE', str(keyfile))
    monkeypatch.setenv('OPENAI_API_KEY', 'paid-openai-secret')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'paid-anthropic-secret')
    route = endpoint(dict(model='gpt-6-luna'))
    assert route.credential() == 'proxy-client-secret'
    assert 'secret' not in json.dumps(route.receipt())
    monkeypatch.setenv('CLIPROXY_API_KEY', 'proxy-env-secret')
    assert route.credential() == 'proxy-env-secret'
    monkeypatch.delenv('CLIPROXY_API_KEY')
    keyfile.unlink()
    with pytest.raises(ValueError, match='client key missing'):
        route.credential()


def test_remote_controller_cannot_accidentally_use_its_own_loopback(monkeypatch):
    monkeypatch.setenv('MODAL_TASK_ID', 'fixture')
    with pytest.raises(ValueError, match='unreachable from Modal'):
        endpoint(dict(model='gpt-6-luna'))
    monkeypatch.setenv('CLIPROXY_BASE_URL', 'https://proxy.example.test/v1')
    assert endpoint(dict(model='gpt-6-luna')).backend == 'cliproxyapi'


@pytest.mark.parametrize('url', ['file:///tmp/key', 'https://user:password@host/v1',
                               'https://host/v1?key=secret', 'https://host/v1#secret'])
def test_endpoint_rejects_credentials_in_urls(url):
    with pytest.raises(ValueError):
        endpoint(dict(provider='openai', model='gpt-6-luna', base_url=url))


def test_mapping_and_parameter_conversion_do_not_substitute_models_or_mutate_config():
    assert proxy_model('anthropic/claude-fable-5-1') == 'claude-fable-5-1'
    assert proxy_model('anthropic/claude-opus-5.5-20260921') == 'claude-opus-5-5-20260921'
    assert proxy_model('openai/gpt-not-in-catalog') == 'gpt-not-in-catalog'
    cfg = dict(model='gpt-6-luna', parameters=dict(reasoning=dict(effort='high'),
               max_tokens=1234, provider=dict(only=['OpenAI']), usage=dict(include=True)))
    route = endpoint(cfg)
    assert parameters(cfg, route) == dict(reasoning_effort='high', max_tokens=1234)
    assert 'provider' in cfg['parameters'] and 'reasoning' in cfg['parameters']
    assert parameters(cfg, route, protocol='responses')['max_output_tokens'] == 1234


@pytest.fixture
def proxy_server(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(dict(path=self.path, body=body, authorization=self.headers.get('Authorization'),
                                 api_key=self.headers.get('x-api-key')))
            if body['model'] == 'missing-model':
                self.send_response(503); self.end_headers(); self.wfile.write(b'server secret must not be logged'); return
            if self.path == '/v1/messages':
                output = dict(content=[dict(type='tool_use', id='tool-1', name='check', input={})])
            elif self.path == '/v1/responses':
                output = dict(output=[dict(type='function_call', call_id='tool-1', name='check', arguments='{}')])
            else:
                output = dict(choices=[dict(finish_reason='tool_calls', message=dict(role='assistant', content=None,
                    tool_calls=[dict(id='tool-1', type='function', function=dict(name='check', arguments='{}'))]))])
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps(output).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    monkeypatch.setenv('CLIPROXY_BASE_URL', f'http://127.0.0.1:{server.server_port}/v1')
    monkeypatch.setenv('CLIPROXY_API_KEY', 'proxy-only-secret')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'must-not-be-used')
    try:
        yield requests
    finally:
        server.shutdown(); server.server_close(); thread.join()


@pytest.mark.parametrize('name,model,path', [
    ('chat', 'openai/gpt-6-luna', '/v1/chat/completions'),
    ('openai', 'openai/gpt-6-luna', '/v1/responses'),
    ('anthropic', 'anthropic/claude-fable-5-1', '/v1/messages'),
])
def test_native_http_tool_roundtrip_through_killable_worker(proxy_server, name, model, path):
    api = provider(dict(provider=name, model=model), 'Use check.',
                   [dict(name='check', description='check', parameters=dict(type='object', properties={}))])
    turn = api.next('start', 5)
    api.result(turn['calls'][0], dict(ok=True))
    api.next(None, 5)
    assert len(proxy_server) == 2 and all(r['path'] == path for r in proxy_server)
    assert all(r['body']['model'] == proxy_model(model) for r in proxy_server)
    assert all((r['api_key'] == 'proxy-only-secret' if name == 'anthropic' else
                r['authorization'] == 'Bearer proxy-only-secret') for r in proxy_server)
    last = proxy_server[-1]['body']
    if name == 'openai':
        assert last['input'][-1]['call_id'] == 'tool-1'
    elif name == 'anthropic':
        assert last['messages'][-1]['content'][0]['tool_use_id'] == 'tool-1'
    else:
        assert last['messages'][-1]['tool_call_id'] == 'tool-1'


def test_proxy_failure_is_sanitized_and_never_falls_back(proxy_server):
    api = provider(dict(provider='chat', model='missing-model'), 'test', [])
    with pytest.raises(RuntimeError, match='model_api_http_503') as error:
        api.next('start', 5)
    assert 'server secret' not in str(error.value) and len(proxy_server) == 1
    assert api.endpoint.backend == 'cliproxyapi'


def test_chat_accepts_null_tool_calls_on_final_text(monkeypatch):
    monkeypatch.setenv('CLIPROXY_API_KEY', 'proxy-secret')
    api = provider(dict(provider='chat', model='gpt-6-luna'), 'check', [])
    output = dict(choices=[dict(message=dict(role='assistant', content='OK', tool_calls=None))])
    with patch('polytopia_bot.skirmish.providers.request', return_value=output):
        assert api.next('start', 5)['calls'] == []


def test_mini_uses_proxy_with_upstream_parser_and_no_cost_gate(monkeypatch):
    from polytopia_bot.skirmish.mini_worker import coding_model
    monkeypatch.setenv('CLIPROXY_API_KEY', 'proxy-only-secret')
    model = coding_model(dict(model='anthropic/claude-fable-5-1', parameters=dict(reasoning=dict(effort='high'))))
    blocks = [dict(type='thinking',thinking='planning',signature='opaque-signature'),
              dict(type='tool_use',id='bash-1',name='bash',input={'command':'printf OK'})]
    output = dict(id='test',model='claude-fable-5-1',stop_reason='tool_use',content=blocks,
                  usage=dict(input_tokens=10,output_tokens=20))
    with patch('polytopia_bot.skirmish.providers.request', return_value=output) as request:
        turn = model.query([model.format_message(role='user', content='check')])
    args = request.call_args.args
    assert args[0] == PROXY_BASE+'/messages' and args[1] == 'proxy-only-secret'
    assert args[2]['model'] == 'claude-fable-5-1' and args[2]['output_config']['effort'] == 'high'
    assert 'usage' not in args[2] and args[2]['tools'][0]['name'] == 'bash'
    assert turn['extra']['actions'][0]['command'] == 'printf OK'
    assert turn['_anthropic_content'] == blocks
    assert request.call_args.kwargs['anthropic'] is True
    assert model.serialize()['info']['inference']['accounting'] == 'subscription_usage'
    assert 'proxy-only-secret' not in json.dumps(model.serialize())
