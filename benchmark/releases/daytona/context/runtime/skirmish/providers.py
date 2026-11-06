"""Provider adapters. Credentials stay in the trusted controller; no built-in web tools."""
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from .inference import endpoint, model_family, parameters


def perform_request(url, key, payload, timeout, *, anthropic=False):
    headers = {'Content-Type': 'application/json'}
    headers.update({'x-api-key': key, 'anthropic-version': '2023-06-01'} if anthropic else
                   {'Authorization': 'Bearer ' + key})
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    from urllib.parse import urlsplit
    # Loopback subscription routing must never traverse the host's external HTTP proxy.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if urlsplit(url).hostname in ('localhost','127.0.0.1','::1') else urllib.request.build_opener()
    try:
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Don't log request/headers or server-provided bodies, which can echo credentials.
        raise RuntimeError(f'model_api_http_{error.code}') from None


class ChatProvider:
    def __init__(self, config, system, tools):
        self.config, self.tools = config, tools
        self.endpoint = endpoint(config)
        self.messages = [{'role': 'system', 'content': system}]

    def next(self, prompt, timeout):
        if prompt:
            self.messages.append({'role': 'user', 'content': prompt})
        cfg = self.config
        route = self.endpoint
        payload = dict(parameters(cfg, route), model=route.model, messages=self.messages,
                       tools=[{'type': 'function', 'function': t} for t in self.tools])
        # No plugins, provider web tools, fallback models or other external tools.
        payload.pop('plugins', None)
        payload.pop('models', None)
        out = request(route.base_url + '/chat/completions', route.credential(),
                      payload, timeout)
        msg = out['choices'][0]['message']
        # Preserve signed/encrypted reasoning_details verbatim through tool turns.
        self.messages.append({k: v for k, v in msg.items()
                              if k in ('role', 'content', 'tool_calls', 'reasoning', 'reasoning_details')})
        calls = [dict(id=t['id'], name=t['function']['name'], arguments=t['function']['arguments'])
                 for t in (msg.get('tool_calls') or [])]
        return dict(raw=out, calls=calls, usage=out.get('usage', {}))

    def result(self, call, result):
        self.messages.append(dict(role='tool', tool_call_id=call['id'], content=json.dumps(result)))


class ResponsesProvider:
    def __init__(self, config, system, tools):
        self.config, self.system = config, system
        self.endpoint = endpoint(config, protocol='responses')
        self.tools = [dict(type='function', **t) for t in tools]
        self.items = []

    def next(self, prompt, timeout):
        if prompt:
            self.items.append(dict(role='user', content=prompt))
        cfg = self.config
        route = self.endpoint
        out = request(route.base_url + '/responses', route.credential(),
                      dict(parameters(cfg, route, protocol='responses'), model=route.model, instructions=self.system,
                           input=self.items, tools=self.tools, store=False), timeout)
        self.items.extend(out['output'])
        calls = [dict(id=x['call_id'], name=x['name'], arguments=x['arguments'])
                 for x in out['output'] if x['type'] == 'function_call']
        return dict(raw=out, calls=calls, usage=out.get('usage', {}))

    def result(self, call, result):
        self.items.append(dict(type='function_call_output', call_id=call['id'], output=json.dumps(result)))


class AnthropicProvider:
    def __init__(self, config, system, tools):
        self.config, self.system = config, system
        self.endpoint = endpoint(config, protocol='anthropic')
        self.tools = [dict(name=t['name'], description=t['description'], input_schema=t['parameters']) for t in tools]
        self.messages = []
        self.pending = []

    def next(self, prompt, timeout):
        if self.pending:
            self.messages.append(dict(role='user', content=self.pending))
            self.pending = []
        if prompt:
            self.messages.append(dict(role='user', content=prompt))
        cfg = self.config
        route = self.endpoint
        out = request(route.base_url + '/messages', route.credential(),
                      dict({'max_tokens': 12000, **parameters(cfg, route, protocol='anthropic')}, model=route.model,
                           system=self.system, messages=self.messages, tools=self.tools), timeout, anthropic=True)
        self.messages.append(dict(role='assistant', content=out['content']))
        calls = [dict(id=x['id'], name=x['name'], arguments=json.dumps(x['input']))
                 for x in out['content'] if x['type'] == 'tool_use']
        return dict(raw=out, calls=calls, usage=out.get('usage', {}))

    def result(self, call, result):
        self.pending.append(dict(type='tool_result', tool_use_id=call['id'], content=json.dumps(result)))


def provider(config, system, tools):
    name = config.get('provider', 'auto')
    if name == 'auto':
        name = model_family(config['model'])
    cls = {'openrouter': ChatProvider, 'cliproxyapi': ChatProvider, 'chat': ChatProvider,
           'responses': ResponsesProvider, 'openai': ResponsesProvider, 'anthropic': AnthropicProvider}.get(name)
    if cls is None:
        raise ValueError('unsupported provider adapter')
    return cls(config, system, tools)


def request(url, key, payload, timeout, *, anthropic=False):
    """An absolute parent-process deadline, including a slowly streaming HTTP body."""
    import subprocess
    import sys
    from urllib.parse import urlsplit
    # The HTTP worker needs no provider credentials in its environment. urllib
    # supports HTTP proxies, not SOCKS; do not inherit an unusable SOCKS setting.
    allowed = ('PATH','PYTHONPATH','LANG','SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE',
               'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY',
               'http_proxy','https_proxy','all_proxy','no_proxy')
    env = {k:v for k,v in os.environ.items() if k in allowed and
           (not k.lower().endswith('_proxy') or k.lower()=='no_proxy' or urlsplit(v).scheme in ('http','https'))}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
    try:
        child = subprocess.run([sys.executable,'-m','polytopia_bot.skirmish.api_worker'],
            input=json.dumps(dict(url=url,key=key,payload=payload,timeout=timeout,anthropic=anthropic)),
            capture_output=True,text=True,timeout=timeout,env=env)
    except subprocess.TimeoutExpired:
        raise TimeoutError('model request deadline') from None
    if child.returncode:
        raise RuntimeError('model API worker exited unexpectedly')
    result = json.loads(child.stdout)
    if not result['ok']:
        raise RuntimeError(result['error'])
    return result['result']
